#!/usr/bin/env python3
"""
agent_bca.py — Brief-Contract Architecture (BCA) for LMAgent. 

Drop this file next to agent_tools.py and agent_llm.py.

─────────────────────────────────────────────────────────────────────
PARADIGM: why this exists
─────────────────────────────────────────────────────────────────────
Naive sub-agent systems dump the parent's full conversation history
into the child's context. On a local model with an 8k window, a
sub-agent spawned at iteration 80 of a complex task gets ~40k tokens
of noise and immediately fails.

Brief-Contract Architecture (BCA) fixes this with four principles:

  1. STRUCTURED BRIEFS (not context dumps)
     Before spawning, the parent writes a brief.json to disk containing
     only: objective, deliverable spec, extracted relevant data, and
     constraints. The child reads only this — not the parent's history.

  2. RESULT CONTRACTS (not TASK_COMPLETE strings)
     Every child writes a result.json before terminating. The parent
     reads structured JSON. No LLM parsing, no string detection fragility.

  3. DEPTH-SCOPED RECURSION
     Every agent carries a depth integer (root=0). Sub-agents can spawn
     sub-agents up to MAX_DEPTH. At max depth, delegate/decompose are
     removed from the tool list entirely — the model cannot attempt them.

  4. SCOPE ISOLATION
     Each agent gets a private scope directory for temp work only.
     Deliverables are ALWAYS written to workspace-root-relative paths.
     Context per agent is O(1) — bounded by brief size, not cumulative history.

─────────────────────────────────────────────────────────────────────
v2 HARDENED — FIXES OVER v1
─────────────────────────────────────────────────────────────────────
  FIX 1 (CRITICAL): initialize_root_agent() — must be called once
    before the root agent loop starts. Root agent now has a proper
    brief. delegate/decompose work identically from root and sub-agents.
    No special-case guards anywhere. _ensure_bca_context() provides a
    safe fallback if initialization was missed.

  FIX 2 (HIGH): Cycle detection in decompose — after Kahn's topological
    sort, validates all tasks were ordered. Cycles produce a clear error
    naming the stuck tasks instead of silently skipping them.

  FIX 3 (HIGH): _auto_write_result artifact scanning — when an agent
    outputs TASK_COMPLETE without calling report_result, we now scan
    the workspace for recently modified files and include them as
    artifacts, so downstream dependency chains aren't broken by empty
    artifact lists.

  FIX 4 (MEDIUM): report_result outside BCA context now returns an
    error instead of silently returning success=True. Agents see the
    failure and can recover rather than thinking they are done.

  FIX 5 (MEDIUM): Config mutation saved/restored per-thread via
    thread-local storage. Values are saved before modification and
    guaranteed to be restored in finally, even if a nested agent also
    mutates them.

  FIX 6 (LOW): _unpack_tc / _parse_tool_args imported directly from
    agent_tools inside _run_bca_agent (lazy, to avoid circular import),
    not indirectly via agent_llm.

  FIX 7 (LOW): cleanup_session_dirs() utility — call at session end to
    remove accumulated .lmagent/agents/* directories.

─────────────────────────────────────────────────────────────────────
PATCH — WORKSPACE PATH FIXES (v2.1)
─────────────────────────────────────────────────────────────────────
  PATCH A (CRITICAL): render_system_context() — scope_path is now
    described as a TEMP/SCRATCH directory only. Deliverable paths are
    explicitly called out as workspace-root-relative. Previously the
    wording "Your working directory is '{scope_path}'" caused sub-agents
    to write all files into the hidden .lmagent/agents/xxx/scope/ dir
    instead of the workspace root where the parent expects them.

  PATCH B (CRITICAL): build_sub_agent_system_prompt() — added an
    explicit FILE WRITING section that repeats: write deliverables to
    the exact path from the brief, NOT into the scope dir. Includes a
    worked example showing a correct vs. incorrect write call.

  PATCH C (HIGH): tool_delegate() — automatically injects a path
    constraint "Write the deliverable to exactly: <path>" whenever
    deliverable_path is non-empty, so the sub-agent has a redundant
    hard constraint on disk even if it skims the brief.

  PATCH D (HIGH): tool_decompose() — same automatic path constraint
    injected for every child task that has a non-empty deliverable path.

  PATCH E (MEDIUM): _run_bca_agent() user turn — the initial user
    message now echoes the deliverable path explicitly so it appears
    near the top of the agent's context, not buried in the system prompt.

  PATCH F (LOW): _scan_recent_artifacts() — now also skips the
    .lmagent/agents/*/scope/ directories so auto-detected artifacts
    never include temp files written to scope dirs. Previously scope
    files could appear in the artifact list and confuse downstream tasks.

─────────────────────────────────────────────────────────────────────
v2.2 FIXES — WORKSPACE DELIVERY RELIABILITY
─────────────────────────────────────────────────────────────────────
  FIX 2.2-A (CRITICAL): Auto-correct condition in _run_bca_agent was
    logically inverted — the guard `not write_path.endswith(deliverable_path)`
    caused the correction to be SKIPPED in the most common failure case
    (agent writes scope/index.html when deliverable IS index.html).
    Fixed to trigger on ANY write to the scope dir unconditionally.

  FIX 2.2-B (HIGH): _dispatch_bca_tool() now accepts an optional `brief`
    parameter and enforces the path redirect as a hard wall at the
    dispatch layer — independent of the loop-level intercept. Double
    protection: loop catches it first, dispatch catches anything that
    slips through.

  FIX 2.2-C (HIGH): verify_and_collect_artifacts() — new public utility
    called by the root agent runner after all sub-agents complete.
    Checks every claimed artifact actually exists on disk. If a file
    landed in the scope dir instead of workspace root, it is physically
    moved/copied to the correct location. Returns a full verification
    report so the root agent has an accurate, confirmed artifact list.

─────────────────────────────────────────────────────────────────────
SCALING PROPERTIES
─────────────────────────────────────────────────────────────────────
  Depth 0 (root): decomposes into N sequential sub-tasks
  Depth 1:        each sub-task can further decompose into M tasks
  Depth 2+:       handles tasks atomically, no further spawning
  Context/agent:  O(1) — always bounded by brief size
  Works on:       any model with ≥4k context window
"""

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_core import (
    Config, Log, Colors, colored, _atomic_write,
    get_current_context, set_current_context,
    TodoManager, PlanManager, TaskStateManager,
    SessionManager, LoopDetector,
    truncate_output,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_DEPTH         = int(os.getenv("AGENT_MAX_DEPTH",   "3"))
BRIEF_DATA_BUDGET = int(os.getenv("BRIEF_DATA_BUDGET", "8000"))

# How many seconds back to scan for artifacts when _auto_write_result fires.
ARTIFACT_SCAN_WINDOW_SECS = int(os.getenv("ARTIFACT_SCAN_WINDOW", "600"))

BRIEF_DIR_NAME  = ".lmagent/agents"
BRIEF_FILENAME  = "brief.json"
RESULT_FILENAME = "result.json"
SCOPE_DIRNAME   = "scope"

RESULT_OK      = "ok"
RESULT_BLOCKED = "blocked"
RESULT_PARTIAL = "partial"
RESULT_ERROR   = "error"


# =============================================================================
# THREAD-LOCAL STATE
# =============================================================================

_bca_local = threading.local()


def get_current_brief() -> Optional["AgentBrief"]:
    return getattr(_bca_local, "brief", None)


def set_current_brief(brief: Optional["AgentBrief"]) -> None:
    _bca_local.brief = brief


def get_brief_manager() -> Optional["BriefManager"]:
    return getattr(_bca_local, "brief_manager", None)


def set_brief_manager(bm: Optional["BriefManager"]) -> None:
    _bca_local.brief_manager = bm


# =============================================================================
# DELIVERABLE SPEC
# =============================================================================

@dataclass
class DeliverableSpec:
    """What the parent expects back from the child."""
    type:        str           # "file" | "data" | "report" | "files" | "task"
    description: str
    path:        str  = ""    # for type="file"
    paths:       List[str] = field(default_factory=list)   # for type="files"
    schema:      str  = ""    # for type="data"
    format:      str  = ""    # "json" | "markdown" | "python" | etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DeliverableSpec":
        return DeliverableSpec(
            type=d.get("type", "file"),
            description=d.get("description", ""),
            path=d.get("path", ""),
            paths=d.get("paths", []),
            schema=d.get("schema", ""),
            format=d.get("format", ""),
        )

    def render_for_prompt(self) -> str:
        lines = [f"Type: {self.type}", f"Description: {self.description}"]
        if self.path:   lines.append(f"File path (write here): {self.path}")
        if self.paths:  lines.append(f"File paths (write here): {', '.join(self.paths)}")
        if self.schema: lines.append(f"Schema: {self.schema}")
        if self.format: lines.append(f"Format: {self.format}")
        return "\n".join(lines)


# =============================================================================
# AGENT BRIEF
# =============================================================================

@dataclass
class AgentBrief:
    """
    The contract handed to an agent before it runs.
    Written to disk as brief.json. Sub-agents read only this —
    never the parent's conversation history.
    Root agent brief is created by initialize_root_agent().
    """
    agent_id:         str
    parent_id:        str
    depth:            int
    max_depth:        int
    objective:        str
    deliverable:      DeliverableSpec
    data:             str          # Extracted relevant data only
    constraints:      List[str]
    parent_objective: str          # Root task breadcrumb
    scope_path:       str          # Agent's private TEMP dir (workspace-relative)
    parent_scope:     str
    session_id:       str
    created_at:       str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["deliverable"] = self.deliverable.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentBrief":
        return AgentBrief(
            agent_id=d["agent_id"],
            parent_id=d.get("parent_id", ""),
            depth=d.get("depth", 0),
            max_depth=d.get("max_depth", MAX_DEPTH),
            objective=d["objective"],
            deliverable=DeliverableSpec.from_dict(d.get("deliverable", {})),
            data=d.get("data", ""),
            constraints=d.get("constraints", []),
            parent_objective=d.get("parent_objective", ""),
            scope_path=d.get("scope_path", ""),
            parent_scope=d.get("parent_scope", ""),
            session_id=d.get("session_id", ""),
            created_at=d.get("created_at", ""),
        )

    def render_system_context(self) -> str:
        depth_label = "ROOT" if self.depth == 0 else f"DEPTH-{self.depth}"
        can_recurse = self.depth < self.max_depth

        lines = [
            f"[AGENT BRIEF — {depth_label}]",
            f"Agent ID     : {self.agent_id}",
            f"Depth        : {self.depth}/{self.max_depth}",
            f"Can recurse  : {'Yes — you may use delegate/decompose tools' if can_recurse else 'No — handle atomically, no sub-delegation'}",
            "",
            "OBJECTIVE:",
            self.objective,
            "",
        ]

        if self.parent_objective and self.parent_objective != self.objective:
            lines += ["BROADER CONTEXT (parent's goal):", self.parent_objective, ""]

        lines += [
            "DELIVERABLE (what you must produce):",
            self.deliverable.render_for_prompt(),
            "",
        ]

        if self.constraints:
            lines += ["CONSTRAINTS (must follow all):"]
            lines += [f"  • {c}" for c in self.constraints]
            lines += [""]

        if self.data.strip():
            lines += [
                "RELEVANT DATA (use this — do not invent data):",
                "─" * 60,
                self.data.strip(),
                "─" * 60,
                "",
            ]

        # PATCH A: Clearly separate scope (temp) from deliverable (workspace-root) paths.
        deliverable_path_note = ""
        if self.deliverable.path:
            deliverable_path_note = (
                f"\n⚠️  DELIVERABLE PATH: Write your output to '{self.deliverable.path}' "
                f"— this is workspace-root-relative (e.g. 'index.html' = workspace root)."
            )
        elif self.deliverable.paths:
            deliverable_path_note = (
                f"\n⚠️  DELIVERABLE PATHS: {', '.join(self.deliverable.paths)} "
                f"— all workspace-root-relative."
            )

        lines += [
            "SCRATCH DIRECTORY (temp files only): " + self.scope_path,
            "  ↳ Use this ONLY for intermediate/temp work, NOT for your deliverable.",
            "WORKSPACE ROOT: All deliverable files must be written relative to workspace root.",
            "  ↳ 'index.html' → workspace_root/index.html",
            "  ↳ 'js/main.js' → workspace_root/js/main.js",
            "  ↳ Do NOT prefix deliverable paths with the scratch directory." + deliverable_path_note,
            "",
            "When done: call report_result with status='ok' and a summary.",
            "If blocked: call report_result with status='blocked' and the reason.",
            "[END BRIEF]",
        ]
        return "\n".join(lines)


# =============================================================================
# AGENT RESULT
# =============================================================================

@dataclass
class AgentResult:
    """Written by the agent to result.json when finished. Parent reads this."""
    agent_id:   str
    status:     str        # RESULT_OK | RESULT_BLOCKED | RESULT_PARTIAL | RESULT_ERROR
    summary:    str
    artifacts:  List[str]  # workspace-relative file paths written
    data:       Any        # structured data if deliverable.type == "data"
    error:      str  = ""
    iterations: int  = 0
    created_at: str  = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentResult":
        return AgentResult(
            agent_id=d.get("agent_id", ""),
            status=d.get("status", RESULT_ERROR),
            summary=d.get("summary", ""),
            artifacts=d.get("artifacts", []),
            data=d.get("data"),
            error=d.get("error", ""),
            iterations=d.get("iterations", 0),
            created_at=d.get("created_at", ""),
        )

    def to_tool_result(self) -> Dict[str, Any]:
        return {
            "success":   self.status in (RESULT_OK, RESULT_PARTIAL),
            "status":    self.status,
            "summary":   self.summary,
            "artifacts": self.artifacts,
            "data":      self.data,
            "error":     self.error,
        }


# =============================================================================
# BRIEF MANAGER
# =============================================================================

class BriefManager:
    """Manages brief.json and result.json lifecycle on disk."""

    def __init__(self, workspace: Path):
        self.workspace  = workspace
        self.agents_dir = workspace / BRIEF_DIR_NAME
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    def _agent_dir(self, agent_id: str) -> Path:
        d = self.agents_dir / agent_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def scope_path(self, agent_id: str) -> Path:
        scope = self._agent_dir(agent_id) / SCOPE_DIRNAME
        scope.mkdir(parents=True, exist_ok=True)
        return scope

    def write_brief(self, brief: AgentBrief) -> Path:
        brief.created_at = datetime.now().isoformat()
        path = self._agent_dir(brief.agent_id) / BRIEF_FILENAME
        _atomic_write(path, brief.to_dict())
        return path

    def read_brief(self, agent_id: str) -> Optional[AgentBrief]:
        path = self._agent_dir(agent_id) / BRIEF_FILENAME
        if not path.exists():
            return None
        try:
            return AgentBrief.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            Log.error(f"[BCA] Failed to read brief for {agent_id}: {e}")
            return None

    def write_result(self, result: AgentResult) -> Path:
        result.created_at = datetime.now().isoformat()
        path = self._agent_dir(result.agent_id) / RESULT_FILENAME
        _atomic_write(path, result.to_dict())
        return path

    def read_result(self, agent_id: str) -> Optional[AgentResult]:
        path = self._agent_dir(agent_id) / RESULT_FILENAME
        if not path.exists():
            return None
        try:
            return AgentResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            Log.error(f"[BCA] Failed to read result for {agent_id}: {e}")
            return None

    def result_exists(self, agent_id: str) -> bool:
        return (self._agent_dir(agent_id) / RESULT_FILENAME).exists()

    def cleanup(self) -> int:
        """Remove all agent directories under .lmagent/agents/. Returns count removed."""
        removed = 0
        if not self.agents_dir.exists():
            return 0
        for d in self.agents_dir.iterdir():
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                    removed += 1
                except Exception as e:
                    Log.warning(f"[BCA] Could not remove agent dir {d.name}: {e}")
        return removed


# =============================================================================
# BRIEF EXTRACTOR
# =============================================================================

class BriefExtractor:
    """
    Extracts the minimum relevant data from a parent's message history
    to populate the brief's 'data' field.

    Heuristic/keyword scoring — no extra LLM calls. Fast, deterministic,
    zero token cost. Prioritises actual data (read results, shell output,
    grep matches) over prose reasoning and bookkeeping.
    """

    HIGH_VALUE_TOOLS  = frozenset({"read", "shell", "grep", "git_diff"})
    STRUCTURAL_TOOLS  = frozenset({"ls", "glob", "git_status"})

    @classmethod
    def extract(
        cls,
        parent_messages: List[Dict[str, Any]],
        objective: str,
        budget: int = BRIEF_DATA_BUDGET,
    ) -> str:
        keywords = cls._extract_keywords(objective)
        scored: List[Tuple[int, str, str]] = []

        for msg in parent_messages:
            if msg.get("role") != "tool":
                continue
            name = msg.get("name", "")
            if name not in cls.HIGH_VALUE_TOOLS and name not in cls.STRUCTURAL_TOOLS:
                continue

            raw = msg.get("content", "")
            try:
                parsed = json.loads(raw)
                if not parsed.get("success"):
                    continue
                text = (
                    parsed.get("content")
                    or parsed.get("output")
                    or parsed.get("stdout")
                    or parsed.get("diff")
                    or ""
                )
                if isinstance(text, list):
                    text = "\n".join(
                        f"{m.get('file','?')}:{m.get('line','?')}: {m.get('content','')}"
                        for m in text[:20]
                    )
                text = str(text).strip()
            except Exception:
                text = str(raw).strip()

            if not text or len(text) < 5:
                continue

            score = sum(1 for kw in keywords if kw.lower() in text.lower())
            score += 10 if name in cls.HIGH_VALUE_TOOLS else 3

            # Recover the path label from the paired assistant tool call
            path_label = ""
            try:
                for m in parent_messages:
                    if m.get("role") != "assistant":
                        continue
                    for tc in (m.get("tool_calls") or []):
                        if tc.get("function", {}).get("name") == name:
                            args = json.loads(tc["function"].get("arguments", "{}"))
                            path_label = args.get("path", "")
                            break
            except Exception:
                pass

            label = f"[{name}{': ' + path_label if path_label else ''}]"
            scored.append((score, label, text))

        scored.sort(key=lambda x: x[0], reverse=True)

        parts: List[str] = []
        used = 0
        for _, label, text in scored:
            if used >= budget:
                break
            per_item = min(budget - used, max(800, budget // max(len(scored), 1)))
            chunk = text[:per_item]
            if len(text) > per_item:
                chunk += f"\n[...truncated {len(text) - per_item} chars]"
            parts.append(f"{label}\n{chunk}")
            used += len(chunk)

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        stop = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "that", "this", "it", "its", "and", "or", "but", "not",
            "you", "i", "we", "they", "create", "write", "make", "build",
            "file", "files", "all", "each", "every",
        }
        words = [
            w.strip(".,;:\"'()[]{}").lower()
            for w in text.split()
            if len(w) > 3
        ]
        return [w for w in words if w not in stop][:20]


# =============================================================================
# SUB-TASK & DECOMPOSITION MANIFEST
# =============================================================================

@dataclass
class SubTask:
    task_id:     str
    objective:   str
    deliverable: DeliverableSpec
    constraints: List[str] = field(default_factory=list)
    depends_on:  List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["deliverable"] = self.deliverable.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SubTask":
        return SubTask(
            task_id=d["task_id"],
            objective=d["objective"],
            deliverable=DeliverableSpec.from_dict(d.get("deliverable", {})),
            constraints=d.get("constraints", []),
            depends_on=d.get("depends_on", []),
        )


# =============================================================================
# DEPTH-SCOPED TOOL FILTERING
# =============================================================================

_LEAF_AGENT_TOOLS = frozenset({
    "read", "write", "edit", "ls", "glob", "grep", "shell",
    "todo_add", "todo_complete", "todo_list",
    "report_result",
})

_INNER_AGENT_EXTRA_TOOLS = frozenset({"delegate", "decompose"})

_ROOT_AGENT_EXTRA_TOOLS = frozenset({
    "mkdir", "git_status", "git_diff", "git_add", "git_commit",
    "task_state_update", "task_state_get",
})


def get_depth_scoped_tools(
    all_schemas: List[Dict[str, Any]],
    depth: int,
    max_depth: int,
) -> List[Dict[str, Any]]:
    """Return the tool schema list appropriate for a given agent depth.

    depth=0 (root) → all tools (root controls its own tool list via get_available_tools)
    depth=1..max_depth-1 → leaf tools + delegate/decompose
    depth=max_depth → leaf tools only (cannot recurse further)
    """
    if depth == 0:
        return all_schemas
    allowed = set(_LEAF_AGENT_TOOLS)
    if depth < max_depth:
        allowed |= _INNER_AGENT_EXTRA_TOOLS
    return [t for t in all_schemas if t["function"]["name"] in allowed]


# =============================================================================
# AGENT ID
# =============================================================================

def make_agent_id(label: str = "") -> str:
    slug = "".join(
        c if c.isalnum() else "_"
        for c in (label or "agent").lower()[:20]
    ).strip("_")
    return f"{slug}_{uuid.uuid4().hex[:6]}"


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================

def build_sub_agent_system_prompt(brief: AgentBrief) -> str:
    can_recurse = brief.depth < brief.max_depth
    depth_rules = (
        "You MAY use 'delegate' to hand off a focused sub-task, or 'decompose' "
        "to split your work into sequential sub-tasks. Use this only when the "
        "task is genuinely too complex to handle atomically."
        if can_recurse else
        "You are at max depth. You MUST handle this task atomically. "
        "The 'delegate' and 'decompose' tools are NOT available to you."
    )

    # PATCH B: Build an unambiguous file-path instruction block.
    if brief.deliverable.path:
        file_path_instruction = (
            f"## FILE PATH — CRITICAL\n\n"
            f"Your deliverable MUST be written to: **`{brief.deliverable.path}`**\n\n"
            f"This is a WORKSPACE-ROOT-RELATIVE path. The tool call must be:\n\n"
            f"    write(path=\"{brief.deliverable.path}\", content=\"...\")\n\n"
            f"❌ WRONG — do NOT write to the scratch dir:\n"
            f"    write(path=\"{brief.scope_path}/{brief.deliverable.path}\", content=\"...\")\n"
            f"    write(path=\"{brief.scope_path}/output.html\", content=\"...\")\n\n"
            f"✅ CORRECT — write directly to the deliverable path:\n"
            f"    write(path=\"{brief.deliverable.path}\", content=\"...\")\n\n"
            f"The scratch directory (`{brief.scope_path}`) is for temp files ONLY.\n"
            f"Your final deliverable must NOT be placed there.\n"
        )
    elif brief.deliverable.paths:
        paths_list = "\n".join(f"  - write(path=\"{p}\", content=\"...\")" for p in brief.deliverable.paths)
        file_path_instruction = (
            f"## FILE PATHS — CRITICAL\n\n"
            f"Your deliverables MUST be written to these workspace-root-relative paths:\n"
            f"{paths_list}\n\n"
            f"Do NOT prefix any of these with the scratch dir (`{brief.scope_path}`).\n"
        )
    else:
        file_path_instruction = (
            f"## FILE PATH — CRITICAL\n\n"
            f"Write all deliverable files to workspace-root-relative paths.\n"
            f"The scratch directory (`{brief.scope_path}`) is for temp files ONLY.\n"
            f"Do NOT write deliverables into the scratch directory.\n"
        )

    return f"""You are a focused execution agent. You have one job: complete the objective in your brief.

{brief.render_system_context()}

{file_path_instruction}

## EXECUTION RULES

**1. Read your brief first.**
Your brief above contains everything you need. Do not ask for more context.

**2. Act immediately.**
No planning prose. Read any referenced files if needed, then execute.

**3. Verify your own work.**
After writing a file: read it back. Does it match the objective?
After a shell command: check exit code and output.

**4. Self-correction (once).**
If something is wrong: fix it once, re-verify, then report.
Same error twice → report_result(status='blocked', ...).

**5. Recursion policy.**
{depth_rules}

**6. Finish with report_result.**
Always end with report_result. Never output TASK_COMPLETE — use report_result instead.
  - Success: report_result(status='ok', summary='...', artifacts=['path/to/file', ...])
  - Blocked: report_result(status='blocked', summary='...', error='exact reason')

**7. Artifacts list.**
In report_result, set artifacts to the ACTUAL workspace-root-relative paths you wrote.
Example: artifacts=['index.html', 'js/main.js', 'css/styles.css']
Never list scratch-dir paths in artifacts.

**8. Data fidelity.**
If your brief's DATA section contains specific values, use ALL of them.
Do not sample, abbreviate, or invent. Missing data → report_result(status='blocked').

## ANTI-PATTERNS
- Do NOT write prose about what you are going to do and then fail to do it.
- Do NOT call report_result and then continue calling tools.
- Do NOT re-read a file you already read unless it changed.
- Do NOT ask clarifying questions. Everything you need is in the brief.
- Do NOT write deliverables into the scratch directory.
"""


# =============================================================================
# FIX 1: ROOT AGENT INITIALIZATION
# =============================================================================

def initialize_root_agent(
    workspace: Path,
    session_id: str,
    task: str,
    messages: List[Dict[str, Any]],
    stream_callback: Optional[Callable] = None,
) -> "AgentBrief":
    """
    Initialize BCA context for the root agent. Call this ONCE before
    the root agent's iteration loop starts, alongside set_tool_context().
    They must always be paired — BCA context and tool context together.

    After this call, get_current_brief() will return a valid root brief
    so that delegate/decompose work identically whether called from the
    root agent or any sub-agent. There are no special cases anywhere.

    Example (in your main agent runner, before the iteration loop):

        set_tool_context(workspace, session_id, todo_mgr, plan_mgr,
                         task_state_mgr, messages, mode=mode,
                         stream_callback=stream_callback)

        initialize_root_agent(workspace=workspace, session_id=session_id,
                               task=task, messages=messages,
                               stream_callback=stream_callback)

        for iteration in range(1, max_iterations + 1):
            response = LLMClient.call(messages, available_tools, ...)
            ...
    """
    bm       = BriefManager(workspace)
    agent_id = make_agent_id("root")

    brief = AgentBrief(
        agent_id=agent_id,
        parent_id="",
        depth=0,
        max_depth=MAX_DEPTH,
        objective=task,
        deliverable=DeliverableSpec(
            type="task",
            description=task,
        ),
        data="",
        constraints=[],
        parent_objective="",
        scope_path=str(bm.scope_path(agent_id).relative_to(workspace)).replace("\\", "/"),
        parent_scope="",
        session_id=session_id,
    )

    bm.write_brief(brief)
    set_current_brief(brief)
    set_brief_manager(bm)

    Log.info(f"[BCA] Root agent initialized: {agent_id} (max_depth={MAX_DEPTH})")
    return brief


def _ensure_bca_context(workspace: Path, objective: str = "") -> Tuple["AgentBrief", "BriefManager"]:
    """
    Guarantee a valid (brief, bm) pair exists on the current thread.

    If initialize_root_agent() was called, this is a no-op returning
    the existing context. If it was NOT called (setup error), this
    creates a root brief on the fly with a warning, so the agent
    degrades gracefully rather than hard-failing.

    Called by tool_delegate and tool_decompose as a safety net.
    """
    brief = get_current_brief()
    bm    = get_brief_manager()

    if brief and bm:
        return brief, bm

    # Setup was missed — recover gracefully with a warning.
    Log.warning(
        "[BCA] BCA context not initialized. "
        "Call initialize_root_agent() before the agent loop. "
        "Creating root brief on demand as fallback."
    )
    ctx         = get_current_context()
    workspace_p = ctx.get("workspace") or workspace
    bm          = BriefManager(workspace_p)
    agent_id    = make_agent_id("root")
    task        = objective or ctx.get("task", "Unknown task")

    brief = AgentBrief(
        agent_id=agent_id,
        parent_id="",
        depth=0,
        max_depth=MAX_DEPTH,
        objective=task,
        deliverable=DeliverableSpec(type="task", description=task),
        data="",
        constraints=[],
        parent_objective="",
        scope_path=str(bm.scope_path(agent_id).relative_to(workspace_p)).replace("\\", "/"),
        parent_scope="",
        session_id=ctx.get("session_id", ""),
    )

    bm.write_brief(brief)
    set_current_brief(brief)
    set_brief_manager(bm)
    return brief, bm


# =============================================================================
# FIX 3: ARTIFACT SCANNER
# =============================================================================

def _scan_recent_artifacts(
    workspace: Path,
    since_seconds: int = ARTIFACT_SCAN_WINDOW_SECS,
) -> List[str]:
    """
    Scan workspace for files modified within the last `since_seconds` seconds.
    Used by _auto_write_result to populate artifacts when an agent outputs
    TASK_COMPLETE without calling report_result.

    PATCH F: Skips hidden directories, .lmagent internals, AND scope dirs
    so temp files written to agent scratch dirs never appear as artifacts.
    """
    cutoff = time.time() - since_seconds
    artifacts: List[str] = []
    # Pre-build the agents dir prefix to filter scope dirs efficiently.
    agents_dir_rel = BRIEF_DIR_NAME  # ".lmagent/agents"
    try:
        for f in workspace.rglob("*"):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(workspace)
            except ValueError:
                continue
            rel_parts = rel.parts
            # Skip hidden dirs and .lmagent internals (includes scope dirs).
            if any(p.startswith(".") for p in rel_parts):
                continue
            # Extra guard: skip anything under .lmagent/agents explicitly.
            rel_str = str(rel).replace("\\", "/")
            if rel_str.startswith(agents_dir_rel):
                continue
            try:
                if f.stat().st_mtime >= cutoff:
                    artifacts.append(rel_str)
            except OSError:
                continue
    except Exception as e:
        Log.warning(f"[BCA] Artifact scan failed: {e}")
    return artifacts


# =============================================================================
# FIX 2.2-C: ARTIFACT VERIFIER
# =============================================================================

def verify_and_collect_artifacts(
    workspace: Path,
    expected_artifacts: List[str],
    brief: "AgentBrief",
    bm: "BriefManager",
) -> Dict[str, Any]:
    """
    Called by the root agent runner after all sub-agents complete.

    For each expected artifact:
      1. Check if it exists at the correct workspace-root-relative path.
      2. If not, scan all agent scope directories to see if it was
         accidentally written there.
      3. If found misplaced, copy it to the correct workspace-root path.
      4. If not found anywhere, report it as missing.

    Returns a dict with:
      found     — list of artifact paths confirmed/recovered in workspace root
      missing   — list of artifact paths not found anywhere
      misplaced — list of artifacts recovered from a scope dir
      all_ok    — True if missing is empty
      summary   — human-readable one-liner
    """
    found:     List[str] = []
    missing:   List[str] = []
    misplaced: List[str] = []

    for art in expected_artifacts:
        art_norm  = art.replace("\\", "/")
        dest_path = workspace / art_norm

        # 1. Happy path — file is exactly where it should be.
        if dest_path.exists():
            found.append(art_norm)
            continue

        # 2. Hunt through every agent's scope dir for a file with the same name.
        recovered = False
        target_name = Path(art_norm).name

        agents_dir = workspace / BRIEF_DIR_NAME
        if agents_dir.exists():
            for agent_dir in agents_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                scope_dir = agent_dir / SCOPE_DIRNAME
                if not scope_dir.exists():
                    continue
                # Search recursively inside the scope dir.
                for candidate in scope_dir.rglob("*"):
                    if not candidate.is_file():
                        continue
                    if candidate.name == target_name:
                        # Found it misplaced — copy to workspace root.
                        try:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(candidate, dest_path)
                            Log.warning(
                                f"[BCA] verify: recovered '{art_norm}' from scope dir "
                                f"'{candidate.relative_to(workspace)}' → workspace root"
                            )
                            misplaced.append(art_norm)
                            found.append(art_norm)
                            recovered = True
                        except Exception as e:
                            Log.error(f"[BCA] verify: could not recover '{art_norm}': {e}")
                        break
                if recovered:
                    break

        if not recovered:
            Log.warning(f"[BCA] verify: '{art_norm}' not found in workspace or any scope dir")
            missing.append(art_norm)

    all_ok  = len(missing) == 0
    summary = f"{len(found)} artifact(s) confirmed in workspace root."
    if misplaced:
        summary += f" {len(misplaced)} recovered from scope dir(s)."
    if missing:
        summary += f" {len(missing)} MISSING: {missing}"

    Log.info(f"[BCA] verify_and_collect_artifacts: {summary}")

    return {
        "found":     found,
        "missing":   missing,
        "misplaced": misplaced,
        "all_ok":    all_ok,
        "summary":   summary,
    }


# =============================================================================
# TOOL: report_result
# =============================================================================

def tool_report_result(
    workspace: Path,
    status: str,
    summary: str,
    artifacts: Optional[List[str]] = None,
    data: Any = None,
    error: str = "",
) -> Dict[str, Any]:
    """
    Agent declares completion. Writes result.json to disk.
    Parent reads this as structured JSON — no string parsing needed.

    FIX 4: Returns an error instead of silently succeeding when called
    outside a BCA context. Agents see the failure and can recover.
    """
    brief = get_current_brief()
    bm    = get_brief_manager()

    # FIX 4: No silent no-op. If context is missing, return a real error.
    if not brief or not bm:
        return {
            "success": False,
            "error": (
                "report_result called outside a BCA agent context. "
                "Root agent should output TASK_COMPLETE, not report_result. "
                "If you are a sub-agent, BCA context was not initialized correctly."
            ),
        }

    valid = {RESULT_OK, RESULT_BLOCKED, RESULT_PARTIAL, RESULT_ERROR}
    if status not in valid:
        return {
            "success": False,
            "error": f"Invalid status '{status}'. Must be one of: {sorted(valid)}",
        }

    # Sanitize artifacts: strip any that accidentally point into scope/agent dirs.
    agents_dir_rel = BRIEF_DIR_NAME
    clean_artifacts: List[str] = []
    for art in (artifacts or []):
        art_norm = art.replace("\\", "/")
        if art_norm.startswith(agents_dir_rel) or art_norm.startswith(".lmagent"):
            Log.warning(
                f"[BCA] report_result: artifact '{art}' points into agent scratch dir — "
                f"removing from list. Write deliverables to workspace root, not scope dir."
            )
        else:
            clean_artifacts.append(art_norm)

    result = AgentResult(
        agent_id=brief.agent_id,
        status=status,
        summary=summary,
        artifacts=clean_artifacts,
        data=data,
        error=error,
        iterations=getattr(_bca_local, "iteration_count", 0),
    )
    bm.write_result(result)
    Log.success(f"[BCA] '{brief.agent_id}' → {status}: {summary[:60]}")

    return {
        "success": status in (RESULT_OK, RESULT_PARTIAL),
        "status":  status,
        "summary": summary,
        "note":    "Result written. Now output TASK_COMPLETE to end this agent.",
    }


# =============================================================================
# TOOL: delegate
# =============================================================================

def tool_delegate(
    workspace: Path,
    objective: str,
    deliverable_type: str,
    deliverable_description: str,
    deliverable_path: str = "",
    deliverable_format: str = "",
    constraints: Optional[List[str]] = None,
    data_hint: str = "",
) -> Dict[str, Any]:
    """
    Spawn a single focused sub-agent to accomplish one objective.
    For splitting into multiple sequential tasks, use 'decompose' instead.

    FIX 1: Uses _ensure_bca_context() so this works from both root
    and sub-agents without any special-case guards.

    PATCH C: Automatically injects an explicit file-path constraint when
    deliverable_path is provided, so the sub-agent has a hard constraint
    in addition to the brief's DELIVERABLE section.
    """
    # FIX 1: Always get a valid context — works from root and sub-agents.
    brief, bm = _ensure_bca_context(workspace, objective)

    if brief.depth >= brief.max_depth:
        return {
            "success": False,
            "error": (
                f"At max depth ({brief.max_depth}). "
                "Handle this task atomically — do not delegate further."
            ),
        }

    ctx      = get_current_context()
    messages = ctx.get("messages") or []

    child_id  = make_agent_id(objective[:25])
    extracted = BriefExtractor.extract(messages, objective)
    if data_hint:
        extracted = f"[Hint from parent]\n{data_hint}\n\n{extracted}"

    deliverable = DeliverableSpec(
        type=deliverable_type,
        description=deliverable_description,
        path=deliverable_path,
        format=deliverable_format,
    )

    scope_rel = str(bm.scope_path(child_id).relative_to(workspace)).replace("\\", "/")

    # PATCH C: Build path-enforcing constraints automatically.
    base_constraints = list(constraints or [])
    if deliverable_path:
        # Normalise path separator for the constraint text.
        norm_path = deliverable_path.replace("\\", "/")
        path_constraints = [
            f"MANDATORY: Write your deliverable to exactly this path: {norm_path}",
            f"This is a workspace-root-relative path. Use: write(path='{norm_path}', content='...')",
            f"Do NOT write the deliverable into your scratch directory.",
            "No placeholders or TODOs anywhere in the file content.",
            f"Verify the file exists after writing: read(path='{norm_path}') or ls(path='.')",
        ]
        # Prepend so they appear first in the constraints block.
        base_constraints = path_constraints + base_constraints

    child_brief = AgentBrief(
        agent_id=child_id,
        parent_id=brief.agent_id,
        depth=brief.depth + 1,
        max_depth=brief.max_depth,
        objective=objective,
        deliverable=deliverable,
        data=extracted,
        constraints=base_constraints,
        parent_objective=brief.objective,
        scope_path=scope_rel,
        parent_scope=brief.scope_path,
        session_id="",
    )

    bm.write_brief(child_brief)
    Log.task(f"[BCA] delegate → '{child_id}' (depth {child_brief.depth}): {objective[:50]}")

    result = _run_bca_agent(
        brief=child_brief,
        workspace=workspace,
        bm=bm,
        parent_ctx=ctx,
        stream_callback=ctx.get("stream_callback"),
    )

    icon = "✓" if result.get("status") == RESULT_OK else "✗"
    Log.info(f"[BCA] {icon} '{child_id}': {result.get('summary', '')[:60]}")

    return {
        "success":   result.get("status") == RESULT_OK,
        "agent_id":  child_id,
        "status":    result.get("status"),
        "summary":   result.get("summary"),
        "artifacts": result.get("artifacts", []),
        "data":      result.get("data"),
        "error":     result.get("error", ""),
    }


# =============================================================================
# TOOL: decompose
# =============================================================================

def tool_decompose(
    workspace: Path,
    manifest_json,          # str OR dict — models pass either
) -> Dict[str, Any]:
    """
    Split the current task into sequential sub-tasks and run them in
    dependency order. Results from completed tasks are injected into the
    brief data of every task that depends on them.

    Improvements over v2:
      - Accepts manifest_json as str OR pre-parsed dict (model-agnostic).
      - Strips markdown fences and trailing commas before JSON parse.
      - Dependency chain halting: if a task fails, all tasks that
        (transitively) depend on it are skipped, not run and failed.
      - Richer dependency injection: includes artifact file contents
        (if small enough) so downstream agents don't need extra reads.
      - Independent tasks that don't depend on the failed task still run.
      - Clearer per-task and aggregate error reporting.

    PATCH D: Automatically injects workspace-root path constraints for
    every child task that has a non-empty deliverable.path, mirroring
    the PATCH C behaviour in tool_delegate.
    """
    # ── Context ───────────────────────────────────────────────────────────────
    brief, bm = _ensure_bca_context(workspace)

    if brief.depth >= brief.max_depth:
        return {
            "success": False,
            "error": (
                f"At max depth ({brief.max_depth}). Cannot decompose further. "
                "Handle this task atomically."
            ),
        }

    # ── Parse manifest (str OR dict) ──────────────────────────────────────────
    try:
        if isinstance(manifest_json, dict):
            manifest_data = manifest_json
        elif isinstance(manifest_json, list):
            # Model passed the tasks array directly — wrap it.
            manifest_data = {"tasks": manifest_json}
        else:
            raw = str(manifest_json).strip()

            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw   = "\n".join(lines[1:])          # drop opening fence line
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3].rstrip()   # drop closing fence

            # Strip trailing commas before ] or } (common model mistake)
            raw = re.sub(r",\s*([}\]])", r"\1", raw)

            manifest_data = json.loads(raw)

        tasks = [SubTask.from_dict(t) for t in manifest_data.get("tasks", [])]
    except Exception as e:
        return {"success": False, "error": f"Could not parse manifest: {e}"}

    # ── Basic validation ───────────────────────────────────────────────────────
    if not tasks:
        return {"success": False, "error": "Manifest contains no tasks."}
    if len(tasks) > 8:
        return {
            "success": False,
            "error": f"Too many tasks ({len(tasks)}). Maximum is 8.",
        }

    task_ids = {t.task_id for t in tasks}

    # Duplicate task_id check
    if len(task_ids) < len(tasks):
        seen: set = set()
        dupes = []
        for t in tasks:
            if t.task_id in seen:
                dupes.append(t.task_id)
            seen.add(t.task_id)
        return {
            "success": False,
            "error": f"Duplicate task_id(s): {dupes}. Every task_id must be unique.",
        }

    # Unknown depends_on references
    for task in tasks:
        bad = [d for d in task.depends_on if d not in task_ids]
        if bad:
            return {
                "success": False,
                "error": (
                    f"Task '{task.task_id}' depends on unknown task(s): {bad}. "
                    f"Valid task IDs: {sorted(task_ids)}"
                ),
            }

        # Self-dependency guard
        if task.task_id in task.depends_on:
            return {
                "success": False,
                "error": f"Task '{task.task_id}' lists itself in depends_on.",
            }

    # ── Topological sort (Kahn's) with cycle detection ────────────────────────
    in_degree   = {t.task_id: len(t.depends_on) for t in tasks}
    by_id       = {t.task_id: t                  for t in tasks}
    queue       = [t for t in tasks if in_degree[t.task_id] == 0]
    ordered: List[SubTask] = []

    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for t in tasks:
            if node.task_id in t.depends_on:
                in_degree[t.task_id] -= 1
                if in_degree[t.task_id] == 0:
                    queue.append(t)

    if len(ordered) < len(tasks):
        ordered_ids = {t.task_id for t in ordered}
        stuck       = sorted(task_ids - ordered_ids)
        return {
            "success": False,
            "error": (
                f"Circular dependency detected — these tasks can never run: {stuck}. "
                "Fix depends_on so the graph is acyclic."
            ),
        }

    Log.task(f"[BCA] decompose: {len(tasks)} sub-tasks for '{brief.agent_id}'")

    # ── Build transitive-blocked set helper ───────────────────────────────────
    def _transitive_blocked(failed_ids: set) -> set:
        """Return all task_ids that (transitively) depend on any failed task."""
        blocked = set(failed_ids)
        changed = True
        while changed:
            changed = False
            for t in tasks:
                if t.task_id not in blocked:
                    if any(d in blocked for d in t.depends_on):
                        blocked.add(t.task_id)
                        changed = True
        return blocked - failed_ids   # exclude the originators themselves

    # ── Execution ─────────────────────────────────────────────────────────────
    ctx      = get_current_context()
    messages = ctx.get("messages") or []

    results:    Dict[str, AgentResult] = {}
    failed_ids: set                    = set()
    skipped:    List[str]              = []

    for task in ordered:
        # Check if this task should be skipped due to a failed dependency.
        blocked_by = [d for d in task.depends_on if d in failed_ids]
        if blocked_by:
            skipped.append(task.task_id)
            Log.warning(
                f"[BCA] skipping '{task.task_id}' — "
                f"blocked by failed dep(s): {blocked_by}"
            )
            # Record a synthetic skipped result so summaries are complete.
            results[task.task_id] = AgentResult(
                agent_id=task.task_id,
                status=RESULT_BLOCKED,
                summary=f"Skipped: dependency failed ({blocked_by})",
                artifacts=[],
                data=None,
                error=f"Upstream failure in: {blocked_by}",
            )
            failed_ids.add(task.task_id)
            continue

        # ── Build dependency data injection ───────────────────────────────────
        dep_parts: List[str] = []
        for dep_id in task.depends_on:
            r = results.get(dep_id)
            if not r:
                continue

            section = [
                f"[Result from dependency '{dep_id}']",
                f"Status  : {r.status}",
                f"Summary : {r.summary}",
            ]

            if r.artifacts:
                section.append(f"Artifacts: {r.artifacts}")
                # Inline small text artifacts so the child doesn't need to read.
                for art_path in r.artifacts[:3]:
                    try:
                        fp = workspace / art_path
                        if fp.is_file() and fp.stat().st_size < 8_000:
                            art_text = fp.read_text(encoding="utf-8", errors="replace")
                            section.append(
                                f"\n--- content of {art_path} ---\n"
                                f"{art_text.strip()}\n"
                                f"--- end {art_path} ---"
                            )
                    except Exception:
                        pass

            if r.data is not None:
                try:
                    section.append(f"Data: {json.dumps(r.data)[:2000]}")
                except Exception:
                    section.append(f"Data: {str(r.data)[:2000]}")

            dep_parts.append("\n".join(section))

        base_data = BriefExtractor.extract(messages, task.objective)
        combined  = "\n\n".join(filter(None, ["\n\n".join(dep_parts), base_data]))

        # ── Spawn child agent ─────────────────────────────────────────────────
        child_id  = make_agent_id(task.task_id)
        scope_rel = str(bm.scope_path(child_id).relative_to(workspace)).replace("\\", "/")

        # PATCH D: Build path-enforcing constraints for this child task.
        base_task_constraints = list(task.constraints)
        if task.deliverable.path:
            norm_path = task.deliverable.path.replace("\\", "/")
            path_constraints = [
                f"MANDATORY: Write your deliverable to exactly this path: {norm_path}",
                f"This is workspace-root-relative. Use: write(path='{norm_path}', content='...')",
                f"Do NOT write the deliverable into your scratch directory.",
                "No placeholders or TODOs anywhere in the file content.",
                f"Verify: read(path='{norm_path}') after writing.",
            ]
            base_task_constraints = path_constraints + base_task_constraints

        child_brief = AgentBrief(
            agent_id=child_id,
            parent_id=brief.agent_id,
            depth=brief.depth + 1,
            max_depth=brief.max_depth,
            objective=task.objective,
            deliverable=task.deliverable,
            data=combined,
            constraints=base_task_constraints + brief.constraints,
            parent_objective=brief.objective,
            scope_path=scope_rel,
            parent_scope=brief.scope_path,
            session_id="",
        )

        bm.write_brief(child_brief)
        Log.task(f"[BCA] sub-task '{task.task_id}' → '{child_id}': {task.objective[:60]}")

        raw_result = _run_bca_agent(
            brief=child_brief,
            workspace=workspace,
            bm=bm,
            parent_ctx=ctx,
            stream_callback=ctx.get("stream_callback"),
        )

        agent_result = AgentResult(
            agent_id=child_id,
            status=raw_result.get("status", RESULT_ERROR),
            summary=raw_result.get("summary", ""),
            artifacts=raw_result.get("artifacts", []),
            data=raw_result.get("data"),
            error=raw_result.get("error", ""),
        )
        results[task.task_id] = agent_result

        icon = "✓" if agent_result.status == RESULT_OK else "✗"
        Log.info(
            f"[BCA] {icon} '{task.task_id}': {agent_result.summary[:80]}"
            + (f" | artifacts: {agent_result.artifacts}" if agent_result.artifacts else "")
        )

        if agent_result.status not in (RESULT_OK, RESULT_PARTIAL):
            failed_ids.add(task.task_id)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ran       = [r for tid, r in results.items() if tid not in skipped]
    all_ok    = all(r.status == RESULT_OK for r in ran) and not skipped
    tasks_ok  = sum(1 for r in ran if r.status == RESULT_OK)

    all_artifacts: List[str] = []
    summaries:     List[str] = []
    for tid in [t.task_id for t in ordered]:
        r = results.get(tid)
        if not r:
            continue
        if tid in skipped:
            summaries.append(f"⊘ {tid}: {r.summary}")
        elif r.status == RESULT_OK:
            summaries.append(f"✓ {tid}: {r.summary}")
            all_artifacts.extend(r.artifacts)
        else:
            summaries.append(f"✗ {tid}: {r.summary} | error: {r.error}")

    if all_ok:
        note = (
            "Decomposition complete — all tasks succeeded. "
            "Call report_result(status='ok', ...) with a summary of everything accomplished."
        )
    elif tasks_ok > 0:
        note = (
            f"{tasks_ok}/{len(ran)} tasks succeeded; {len(skipped)} skipped. "
            "Call report_result(status='partial', ...) describing what worked and what failed."
        )
    else:
        note = (
            "All tasks failed or were skipped. "
            "Call report_result(status='blocked', ...) explaining the root cause."
        )

    return {
        "success":       all_ok,
        "tasks_run":     len(ran),
        "tasks_ok":      tasks_ok,
        "tasks_skipped": len(skipped),
        "artifacts":     all_artifacts,
        "summaries":     summaries,
        "note":          note,
    }


# =============================================================================
# CORE BCA EXECUTION LOOP
# =============================================================================

def _run_bca_agent(
    brief: AgentBrief,
    workspace: Path,
    bm: BriefManager,
    parent_ctx: Dict[str, Any],
    stream_callback: Optional[Callable[[str], None]] = None,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a single BCA agent from its brief.

    Key differences from old run_sub_agent:
    - Context is brief.render_system_context(), never parent history.
    - Tools are depth-scoped (leaf agents cannot recurse).
    - Completion is signalled by report_result tool call, not string detection.
    - Result is read from result.json — structured, no parsing.
    - Loop detector is tighter — sub-agents do less work per agent.

    FIX 5: Config mutation uses thread-local save/restore so nested
    agents on the same thread correctly restore to their own prior values.
    FIX 6: _unpack_tc/_parse_tool_args imported from agent_tools directly.
    PATCH E: User turn echoes deliverable path explicitly near top of context.
    FIX 2.2-A: Auto-correct now triggers on ANY write to the scope dir,
    not just when the filename doesn't match the deliverable path.
    """
    # FIX 6: Lazy imports — avoids circular dependency at module load time.
    # Import from agent_tools directly (where these are defined), not via agent_llm.
    from agent_tools import (  # noqa: PLC0415
        TOOL_SCHEMAS,
        TOOL_HANDLERS,
        set_tool_context,
        _unpack_tc,
        _parse_tool_args,
    )
    from agent_llm import LLMClient  # noqa: PLC0415

    max_iter = max_iterations or max(
        8, Config.MAX_SUB_AGENT_ITERATIONS - brief.depth * 4
    )

    session_mgr = SessionManager(workspace)
    session_id  = session_mgr.create(
        f"[BCA-d{brief.depth}] {brief.objective[:60]}",
        parent_session=parent_ctx.get("session_id", ""),
    )
    brief.session_id = session_id

    # PATCH E: Build user turn that explicitly echoes the deliverable path
    # so it appears near the top of the model's effective context window.
    if brief.deliverable.path:
        user_turn = (
            f"Execute your brief now.\n\n"
            f"Objective: {brief.objective}\n\n"
            f"⚠️  WRITE YOUR DELIVERABLE TO: {brief.deliverable.path}\n"
            f"Use: write(path='{brief.deliverable.path}', content='...')\n"
            f"Do NOT write to your scratch directory ({brief.scope_path}).\n\n"
            f"When done, call report_result with artifacts=['{brief.deliverable.path}']. "
            f"Do not ask questions."
        )
    elif brief.deliverable.paths:
        paths_str = ", ".join(f"'{p}'" for p in brief.deliverable.paths)
        user_turn = (
            f"Execute your brief now.\n\n"
            f"Objective: {brief.objective}\n\n"
            f"⚠️  WRITE YOUR DELIVERABLES TO: {', '.join(brief.deliverable.paths)}\n"
            f"All paths are workspace-root-relative. "
            f"Do NOT write to your scratch directory ({brief.scope_path}).\n\n"
            f"When done, call report_result with artifacts=[{paths_str}]. "
            f"Do not ask questions."
        )
    else:
        user_turn = (
            f"Execute your brief now.\n\n"
            f"Objective: {brief.objective}\n\n"
            f"Write all deliverable files to workspace-root-relative paths "
            f"(NOT into your scratch directory at {brief.scope_path}).\n\n"
            f"When done, call report_result. Do not ask questions."
        )

    messages = [
        {"role": "system", "content": build_sub_agent_system_prompt(brief)},
        {"role": "user",   "content": user_turn},
    ]

    # Depth-scoped tool list.
    depth_tools = get_depth_scoped_tools(TOOL_SCHEMAS, brief.depth, brief.max_depth)
    bca_schemas = _get_bca_tool_schemas(brief.depth, brief.max_depth)
    existing    = {t["function"]["name"] for t in depth_tools}
    for s in bca_schemas:
        if s["function"]["name"] not in existing:
            depth_tools.append(s)
            existing.add(s["function"]["name"])

    todo_mgr       = TodoManager(workspace, session_id)
    plan_mgr       = PlanManager(workspace, session_id)
    task_state_mgr = TaskStateManager(workspace, session_id)

    # Save entire parent context so we can restore it in finally.
    prev_ctx   = get_current_context()
    prev_brief = get_current_brief()
    prev_bm    = get_brief_manager()

    set_current_brief(brief)
    set_brief_manager(bm)
    set_tool_context(
        workspace, session_id,
        todo_mgr, plan_mgr, task_state_mgr,
        messages, mode="interactive",
        stream_callback=stream_callback,
    )

    # FIX 5: Save Config values into thread-local so nested agents restore
    # the values THEY saved, not the values of a different nesting level.
    saved_streak   = Config.MAX_SAME_TOOL_STREAK
    saved_progress = Config.MAX_NO_PROGRESS_ITERS
    saved_errors   = Config.MAX_ERRORS

    Config.MAX_SAME_TOOL_STREAK  = max(3, 5 - brief.depth)
    Config.MAX_NO_PROGRESS_ITERS = max(5, 9 - brief.depth)
    Config.MAX_ERRORS            = 3

    detector                    = LoopDetector()
    _bca_local.iteration_count  = 0

    # Pre-compute normalised scope prefix once for the whole loop.
    scope_norm = brief.scope_path.rstrip("/").replace("\\", "/")

    try:
        for iteration in range(1, max_iter + 1):
            _bca_local.iteration_count = iteration

            loop_msg = detector.check(iteration)
            if loop_msg:
                Log.warning(f"[BCA] '{brief.agent_id}' loop detected: {loop_msg}")
                return _error_result(brief, bm, f"Loop detected: {loop_msg}", iteration)

            response = LLMClient.call(
                messages, depth_tools, stream_callback=stream_callback
            )
            if "error" in response:
                return _error_result(
                    brief, bm, f"LLM error: {response['error']}", iteration
                )

            content    = response.get("content", "")
            tool_calls = response.get("tool_calls") or []

            if not content and not tool_calls:
                detector.track_empty()
                continue

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content:    assistant_msg["content"]    = content
            if tool_calls: assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            result_written = False

            for tc in tool_calls:
                fn_name, args_raw, tc_id = _unpack_tc(tc, f"tc_{iteration}")
                args, err = _parse_tool_args(fn_name, args_raw)

                if err:
                    detector.track_error()
                    messages.append({
                        "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                        "content": json.dumps({"success": False, "error": err}),
                    })
                    continue

                # ── FIX 2.2-A: Hard path intercept (loop layer) ───────────────
                # Trigger on ANY write() whose path starts with the scope dir,
                # regardless of whether the filename matches the deliverable.
                # The old condition `not write_path.endswith(deliverable_path)`
                # was logically wrong — it SKIPPED the correction in the exact
                # case where the agent wrote scope/index.html and deliverable
                # IS index.html.
                if fn_name == "write" and scope_norm:
                    write_path = args.get("path", "").replace("\\", "/")
                    if write_path.startswith(scope_norm + "/"):
                        # Determine the corrected path.
                        if brief.deliverable.path:
                            corrected = brief.deliverable.path
                        else:
                            # Strip scope prefix, keep the relative remainder.
                            corrected = write_path[len(scope_norm):].lstrip("/")
                        Log.warning(
                            f"[BCA] Auto-corrected write: "
                            f"'{write_path}' → '{corrected}' "
                            f"(agent '{brief.agent_id}' tried to write to scope dir)"
                        )
                        args = dict(args)
                        args["path"] = corrected

                # Pass brief to dispatch for the second layer of enforcement.
                result  = _dispatch_bca_tool(fn_name, args, workspace, brief=brief)
                success = result.get("success", False)
                detector.track_tool(fn_name, args, success)
                if success: detector.track_success(iteration)
                else:       detector.track_error()

                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                    "content": json.dumps(result),
                })

                if fn_name == "report_result":
                    result_written = True
                    break   # Stop processing further tool calls this turn.

            if result_written:
                break

            # Secondary completion: agent output TASK_COMPLETE without report_result.
            if content and "TASK_COMPLETE" in content.upper():
                if not bm.result_exists(brief.agent_id):
                    _auto_write_result(brief, bm, workspace, content, iteration)
                break

        # Read the structured result from disk.
        final = bm.read_result(brief.agent_id)
        if final:
            return final.to_tool_result()

        # Agent exited without writing any result.
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg["content"]
                break

        return {
            "success":   False,
            "status":    RESULT_ERROR,
            "summary":   "Agent exited without calling report_result",
            "error":     last_content[:300] if last_content else "No output produced",
            "artifacts": [],
            "data":      None,
        }

    finally:
        # FIX 5: Always restore from THIS frame's saved values, never from
        # a different thread or nesting level.
        Config.MAX_SAME_TOOL_STREAK  = saved_streak
        Config.MAX_NO_PROGRESS_ITERS = saved_progress
        Config.MAX_ERRORS            = saved_errors
        # Restore entire parent BCA + tool context.
        set_current_brief(prev_brief)
        set_brief_manager(prev_bm)
        set_current_context(prev_ctx)


def _dispatch_bca_tool(
    fn_name: str,
    args: Dict[str, Any],
    workspace: Path,
    brief: Optional["AgentBrief"] = None,
) -> Dict[str, Any]:
    """Route a tool call to the correct handler.

    BCA tools (report_result, delegate, decompose) take priority so they
    always use the BCA implementations, never any stale TOOL_HANDLERS entry.

    FIX 2.2-B: Accepts an optional `brief` parameter and enforces a hard
    path redirect at the dispatch layer as a second line of defence,
    independent of the loop-level intercept in _run_bca_agent.
    Any write() that still targets the scope dir after the loop intercept
    is silently redirected here before the actual write tool runs.
    """
    from agent_tools import TOOL_HANDLERS  # noqa: PLC0415

    # FIX 2.2-B: Dispatch-layer scope-dir enforcement.
    if fn_name == "write" and brief and brief.scope_path:
        write_path = args.get("path", "").replace("\\", "/")
        scope_norm = brief.scope_path.rstrip("/").replace("\\", "/")
        if write_path.startswith(scope_norm + "/"):
            corrected = (
                brief.deliverable.path
                if brief.deliverable.path
                else write_path[len(scope_norm):].lstrip("/")
            )
            Log.warning(
                f"[BCA] _dispatch: redirecting write '{write_path}' → '{corrected}' "
                f"(scope-dir write slipped past loop intercept)"
            )
            args = dict(args)
            args["path"] = corrected

    bca_handlers: Dict[str, Callable] = {
        "report_result": tool_report_result,
        "delegate":      tool_delegate,
        "decompose":     tool_decompose,
    }
    handler = bca_handlers.get(fn_name) or TOOL_HANDLERS.get(fn_name)
    if not handler:
        return {"success": False, "error": f"Unknown tool: {fn_name}"}
    try:
        return handler(workspace, **args)
    except Exception as e:
        Log.error(f"[BCA] Tool '{fn_name}' raised exception: {e}")
        return {"success": False, "error": str(e)}


def _error_result(
    brief: AgentBrief,
    bm: BriefManager,
    error: str,
    iteration: int,
) -> Dict[str, Any]:
    r = AgentResult(
        agent_id=brief.agent_id,
        status=RESULT_ERROR,
        summary=f"Agent failed: {error[:100]}",
        artifacts=[],
        data=None,
        error=error,
        iterations=iteration,
    )
    bm.write_result(r)
    return r.to_tool_result()


def _auto_write_result(
    brief: AgentBrief,
    bm: BriefManager,
    workspace: Path,
    content: str,
    iteration: int,
) -> None:
    """
    Fallback result writer when an agent outputs TASK_COMPLETE without
    calling report_result.

    FIX 3 + PATCH F: Scans workspace for recently modified files, excluding
    scope/agent scratch dirs, so downstream dependency chains have accurate
    artifact lists.
    """
    is_blocked = "BLOCKED" in content.upper()
    artifacts  = [] if is_blocked else _scan_recent_artifacts(workspace)

    r = AgentResult(
        agent_id=brief.agent_id,
        status=RESULT_BLOCKED if is_blocked else RESULT_OK,
        summary=content[:200].strip(),
        artifacts=artifacts,
        data=None,
        error=content[:200] if is_blocked else "",
        iterations=iteration,
    )
    bm.write_result(r)
    if artifacts:
        Log.info(f"[BCA] _auto_write_result: inferred {len(artifacts)} artifact(s): {artifacts}")


# =============================================================================
# BACKWARD-COMPAT: tool_task → BCA
# =============================================================================

def tool_task_bca(
    workspace: Path,
    task_type: str,
    instructions: str,
    file_path: str,
) -> Dict[str, Any]:
    """
    Drop-in BCA replacement for tool_task. Same 4-arg signature.

    FIX 1: Uses get_current_brief() now that root is always initialized.
    No more inline brief construction — consistent with delegate/decompose.
    Path constraints are injected just like tool_delegate (PATCH C).
    """
    if not Config.ENABLE_SUB_AGENTS:
        return {"success": False, "error": "Sub-agents disabled"}

    objective     = f"Create {task_type} file at '{file_path}': {instructions[:200]}"
    current_brief, bm = _ensure_bca_context(workspace, objective)

    ctx      = get_current_context()
    messages = ctx.get("messages") or []

    agent_id  = make_agent_id(file_path.replace("/", "_").replace(".", "_")[:20])
    scope_rel = str(bm.scope_path(agent_id).relative_to(workspace)).replace("\\", "/")
    norm_path = file_path.replace("\\", "/")

    brief = AgentBrief(
        agent_id=agent_id,
        parent_id=current_brief.agent_id,
        depth=current_brief.depth + 1,
        max_depth=current_brief.max_depth,
        objective=objective,
        deliverable=DeliverableSpec(
            type="file",
            description=instructions[:300],
            path=norm_path,
            format=task_type,
        ),
        data=BriefExtractor.extract(messages, objective),
        constraints=[
            f"MANDATORY: Write the file to exactly this path: {norm_path}",
            f"This is workspace-root-relative. Use: write(path='{norm_path}', content='...')",
            "Do NOT write the deliverable into your scratch directory.",
            "No placeholders or TODOs anywhere in the file content.",
            f"Verify the file exists after writing: read(path='{norm_path}')",
        ],
        parent_objective=current_brief.objective,
        scope_path=scope_rel,
        parent_scope=current_brief.scope_path,
        session_id="",
    )

    bm.write_brief(brief)
    Log.task(f"[BCA] task_bca: '{agent_id}' depth={brief.depth} → {norm_path}")

    result = _run_bca_agent(
        brief=brief,
        workspace=workspace,
        bm=bm,
        parent_ctx=ctx,
        stream_callback=ctx.get("stream_callback"),
    )

    return {
        "success":   result.get("success", False),
        "file_path": norm_path,
        "agent_id":  agent_id,
        "status":    result.get("status"),
        "summary":   result.get("summary"),
        "artifacts": result.get("artifacts", []),
        "error":     result.get("error", ""),
    }


# =============================================================================
# FIX 7: SESSION CLEANUP UTILITY
# =============================================================================

def cleanup_session_dirs(workspace: Path) -> int:
    """
    Remove all .lmagent/agents/* directories created during this session.
    Call this at the end of an agent session to keep the workspace clean.
    Returns the number of agent directories removed.

    Example:
        # At the end of your main agent runner:
        removed = cleanup_session_dirs(workspace)
        Log.info(f"Cleaned up {removed} BCA agent directories.")
    """
    bm = BriefManager(workspace)
    removed = bm.cleanup()
    Log.info(f"[BCA] cleanup_session_dirs: removed {removed} agent dir(s)")
    return removed


# =============================================================================
# BCA TOOL SCHEMAS
# =============================================================================

_BCA_SCHEMAS_ALL: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "report_result",
        "description": (
            "Report task completion. Call this when your work is done or when you are blocked. "
            "Always the LAST tool call you make. Never call tools after this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["ok", "blocked", "partial", "error"],
                    "description": (
                        "'ok'=fully complete, 'blocked'=cannot proceed, "
                        "'partial'=some done but not all, 'error'=unexpected failure"
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": "1-3 sentences: what you accomplished and what files you produced.",
                },
                "artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Workspace-root-relative paths of every deliverable file you wrote. "
                        "Example: ['index.html', 'css/styles.css', 'js/main.js']. "
                        "Do NOT include scratch-dir paths. "
                        "Always populate this — downstream agents depend on it."
                    ),
                },
                "data": {
                    "description": (
                        "Structured data if deliverable.type was 'data'. "
                        "null for file deliverables."
                    ),
                },
                "error": {
                    "type": "string",
                    "description": "If status is 'blocked' or 'error': the specific reason why.",
                },
            },
            "required": ["status", "summary"],
        },
    }},

    {"type": "function", "function": {
        "name": "delegate",
        "description": (
            "Spawn a focused sub-agent to accomplish one specific objective with a clear deliverable. "
            "Use for a single well-defined atomic task. "
            "For multiple sequential tasks with dependencies, use 'decompose' instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "One precise sentence: exactly what the sub-agent must accomplish.",
                },
                "deliverable_type": {
                    "type": "string",
                    "enum": ["file", "data", "report", "files"],
                    "description": "The kind of output expected.",
                },
                "deliverable_description": {
                    "type": "string",
                    "description": "Specific description of the expected output. Be precise.",
                },
                "deliverable_path": {
                    "type": "string",
                    "description": (
                        "Workspace-root-relative file path if deliverable_type is 'file'. "
                        "Example: 'index.html', 'src/app.js', 'css/styles.css'. "
                        "Do NOT prefix with scratch or agent dirs."
                    ),
                },
                "deliverable_format": {
                    "type": "string",
                    "description": "Format hint: 'python', 'json', 'html', 'markdown', etc.",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hard rules the sub-agent must follow. Be specific.",
                },
                "data_hint": {
                    "type": "string",
                    "description": (
                        "Specific data, specs, or context the sub-agent needs "
                        "that is not already in workspace files. "
                        "Include colors, URLs, API responses, or any other facts."
                    ),
                },
            },
            "required": ["objective", "deliverable_type", "deliverable_description"],
        },
    }},

    {"type": "function", "function": {
        "name": "decompose",
        "description": (
            "Split your task into sequential sub-tasks and execute them in dependency order. "
            "Each task's artifacts and results are automatically passed to dependent tasks. "
            "Maximum 8 tasks. No circular dependencies. "
            "Use when tasks must happen in sequence and later tasks need earlier results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "manifest_json": {
                    "type": "string",
                    "description": (
                        'Valid JSON string: '
                        '{"tasks": [{'
                        '"task_id": "t1", '
                        '"objective": "precise description of what this task does", '
                        '"deliverable": {'
                        '"type": "file", '
                        '"description": "...", '
                        '"path": "index.html"'
                        '}, '
                        '"constraints": ["rule 1"], '
                        '"depends_on": []'
                        '}]}. '
                        "IMPORTANT: deliverable.path must be workspace-root-relative "
                        "(e.g. 'index.html', 'css/styles.css', 'js/main.js'). "
                        "Do NOT include scope or agent-dir prefixes in paths. "
                        "task_id must be unique. depends_on lists task_ids that must complete first. "
                        "No circular dependencies. Maximum 8 tasks."
                    ),
                },
            },
            "required": ["manifest_json"],
        },
    }},
]


def _get_bca_tool_schemas(depth: int, max_depth: int) -> List[Dict[str, Any]]:
    """Return BCA tool schemas appropriate for the given depth.

    report_result: always available (every agent needs to signal completion).
    delegate, decompose: only available when depth < max_depth.
    """
    always  = {"report_result"}
    recurse = {"delegate", "decompose"}
    return [
        s for s in _BCA_SCHEMAS_ALL
        if s["function"]["name"] in always
        or (depth < max_depth and s["function"]["name"] in recurse)
    ]
