#!/usr/bin/env python3
"""
agent_llm.py — LLM client, agent execution loops, and system prompts.

Imports tool definitions from agent_tools.  agent_tools lazy-imports
run_sub_agent / SUB_AGENT_SYSTEM_PROMPT from here inside tool_task() to
avoid a circular module-level dependency.

Dependency graph (no cycles):
    agent_core
        ↑
    agent_tools   ←── (call-time lazy import only) ──┐
        ↑                                             │
    agent_llm  ────────────────────────────────────────
"""

import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from agent_core import (
    Config, Colors, Log, PermissionMode,
    truncate_output, strip_thinking,
    get_current_context, set_current_context, _get_ctx,
    TodoManager, PlanManager, TaskStateManager,
    SessionManager, LoopDetector,
    colored,
)
from agent_tools import (
    TOOL_SCHEMAS,
    TOOL_HANDLERS,
    get_available_tools,
    _REQUIRED_ARG_TOOLS,
    _unpack_tc,
    _parse_tool_args,
    set_tool_context,
)

# Thread-local storage for per-session LLM flags (e.g. tool_choice rejection).
_llm_local = threading.local()

# Names of tools that sub-agents are allowed to call.
# shell, vision, git are intentionally excluded — sub-agents are file-only.
_SUB_AGENT_TOOL_NAMES = frozenset({"write", "read", "edit", "ls", "glob"})

# Names of tools available during plan-mode (read-only orientation).
_PLAN_TOOL_NAMES = frozenset({"ls", "read", "glob", "grep", "git_status"})


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a skilled, proactive digital coworker with real tools that make real changes.

{soul_section}

## PRIME DIRECTIVE
Call a tool every iteration, or output TASK_COMPLETE. Nothing else is valid output.

---

## TASK_COMPLETE
Output TASK_COMPLETE the instant the deliverable exists and is verified. It is a full stop — nothing follows it.

When to output it:
- File/output exists and is verified correct
- todo_complete returned all_complete: true
- The user's question is answered
- The plan is finished and verified

Do NOT summarize, narrate, re-verify, or explain after TASK_COMPLETE.

---

## RULES

**1. Stop the moment the job is done.**
Deliverable verified → TASK_COMPLETE. No exceptions.

**2. Verify once.**
After write/edit → verify with read or ls exactly once.
Pass → TASK_COMPLETE. Fail → one fix attempt → TASK_COMPLETE or BLOCKED.

**3. All means all — count first.**
"All files" → enumerate with glob/ls, process each one, then TASK_COMPLETE.

**4. Execute directly.**
Make changes directly. Only write a script if the task IS to produce a script.

**5. Todos are bookkeeping only.**
- Complete each todo ID exactly once — never twice
- all_complete: true → TASK_COMPLETE immediately
- Never add todos after execution starts

**6. Self-correct once.**
Error → read it → adapt → one retry with a different approach.
Same error twice → BLOCKED.

**7. No redundant work.**
Already succeeded → skip. Already verified → stop. Done → TASK_COMPLETE.

---

## LOOP PREVENTION
Before every tool call, ask:
1. Did this exact operation already succeed? → Skip it.
2. Is the deliverable verified? → TASK_COMPLETE.
3. Am I about to complete an already-completed todo? → TASK_COMPLETE.
4. Have I done 3+ iterations with no new output? → TASK_COMPLETE or BLOCKED.

---

## TOOL CALL RULES
Every call needs ALL required arguments:

| Tool   | Required args |
|--------|---------------|
| write  | path + complete content (no placeholders) |
| read   | path |
| edit   | path, search, replace |
| shell  | command |
| vision | path (workspace-relative image file) |

Unknown path? → ls or glob first, then act.

---

## WORKFLOW
1. **Orient** — glob/ls/read to understand scope (skip if task is obvious)
2. **Execute** — act and verify each step once
3. **Confirm** — deliverable exists and is correct (one check)
4. **Done** — TASK_COMPLETE

Single-file task? Skip step 1. Write → verify → TASK_COMPLETE.

---

## SCHEDULED WAIT
To pause until a future time:

  WAIT: <ISO_datetime>: <reason>
  Example: WAIT: 2026-03-01T09:00:00: Waiting for market open.

Do NOT follow WAIT with TASK_COMPLETE.

---

## BLOCKED FORMAT
When stuck, output exactly:

  BLOCKED:
  Reason: <one sentence>
  Failed operation: <tool name>
  Error received: <exact error>
  What I need: <specific requirement>

---

## TOOLS
Files:      read, write, edit, glob, grep, ls, mkdir
Shell:      shell (sandboxed — 30s timeout, 512MB memory default)
Git:        git_status, git_diff, git_add, git_commit, git_branch
Todos:      todo_add, todo_complete, todo_update, todo_list
Task State: task_state_update, task_state_get, task_reconcile
Planning:   plan_complete_step
Delegation: task (sub-agent, one per file maximum)
Vision:     vision (only present when a VLM is loaded — use for any image/screenshot task)
"""

SUB_AGENT_SYSTEM_PROMPT = """You are a precise file-creation agent. One job: create the file exactly as specified.

Rules:
1. Two sentences of reasoning maximum — then write immediately.
2. write requires "path" AND complete "content" — no placeholders ever.
3. After writing: read it back once to verify.
4. Verified correct → TASK_COMPLETE
5. Wrong or failed → BLOCKED: <one line reason>

Never loop. Never ask questions. Never plan. TASK_COMPLETE means stop — nothing follows it.

Tools: write, read, edit, ls, glob
"""

PLAN_MODE_PROMPT = """You are in PLAN MODE. Produce a concrete, actionable plan. Do not execute anything.

Think briefly (3 sentences max), then output the JSON immediately.

Ask yourself:
- What is actually in scope?
- What is the correct order and what has dependencies?
- Where are the failure risks?
- How is each step verified as complete?

OUTPUT FORMAT — valid JSON only, no text before or after:

{
  "title": "Short descriptive title",
  "goal": "One sentence: what does done look like?",
  "risk_areas": ["Risk 1", "Risk 2"],
  "steps": [
    {
      "id": "step_1",
      "description": "Concrete action — specific enough to execute without clarification",
      "verification": "Exact check that proves this step is complete",
      "risk": "low|medium|high",
      "dependencies": []
    }
  ]
}

After the JSON, output exactly:
PLAN_APPROVED
"""


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    _HEADERS: Dict[str, str] = {"Content-Type": "application/json"}

    @classmethod
    def _headers(cls) -> Dict[str, str]:
        return {**cls._HEADERS, "Authorization": f"Bearer {Config.LLM_API_KEY}"}

    @classmethod
    def _tool_choice_rejected(cls) -> bool:
        return getattr(_llm_local, "tool_choice_rejected", False)

    @classmethod
    def _set_tool_choice_rejected(cls, val: bool) -> None:
        _llm_local.tool_choice_rejected = val

    @staticmethod
    def validate_connection() -> Optional[str]:
        try:
            r = requests.post(
                Config.LLM_URL,
                json={"messages": [{"role": "user", "content": "test"}], "max_tokens": 1},
                headers=LLMClient._headers(), timeout=10,
            )
            if r.status_code == 200: return None
            if r.status_code == 401: return "Authentication failed"
            return f"HTTP {r.status_code}"
        except requests.ConnectionError:
            return f"Cannot connect to {Config.LLM_URL}"
        except Exception as e:
            return f"Connection error: {e}"

    @classmethod
    def _build_payload(cls, messages: List[Dict[str, Any]],
                        tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages":    messages,
            "temperature": Config.TEMPERATURE,
            "stream":      True,
        }
        if Config.LLM_MODEL:      payload["model"]      = Config.LLM_MODEL
        if Config.MAX_TOKENS > 0: payload["max_tokens"] = Config.MAX_TOKENS
        if Config.THINKING_MODEL:
            payload["max_tokens"] = max(payload.get("max_tokens", 0) or 0,
                                        Config.THINKING_MAX_TOKENS)
            if payload.get("temperature", 0) > 0.7:
                Log.warning("High temperature with thinking model — consider TEMPERATURE<=0.6")
        if tools:
            payload["tools"] = tools
            if not cls._tool_choice_rejected():
                payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _wait_for_server(max_wait: int = 60) -> bool:
        Log.info(f"Waiting for LLM server (up to {max_wait}s)…")
        deadline, interval = time.time() + max_wait, 3
        while time.time() < deadline:
            time.sleep(interval)
            try:
                r = requests.post(
                    Config.LLM_URL,
                    json={"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                    headers=LLMClient._headers(), timeout=5,
                )
                if r.status_code in (200, 400, 401):
                    Log.success("LLM server back online.")
                    return True
            except Exception:
                pass
            interval = min(interval + 2, 10)
        Log.error("LLM server did not recover in time.")
        return False

    @classmethod
    def call(cls, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        payload    = cls._build_payload(messages, tools)
        last_error: Optional[str] = None
        for attempt in range(1, Config.LLM_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    Config.LLM_URL, json=payload, headers=cls._headers(),
                    stream=True, timeout=Config.LLM_TIMEOUT,
                )
                if resp.status_code == 400 and "tool_choice" in (resp.text or "").lower():
                    Log.warning("Model rejected tool_choice — disabling for this session")
                    cls._set_tool_choice_rejected(True)
                    payload.pop("tool_choice", None)
                    resp = requests.post(
                        Config.LLM_URL, json=payload, headers=cls._headers(),
                        stream=True, timeout=Config.LLM_TIMEOUT,
                    )
                if resp.status_code in (500, 503):
                    Log.warning(f"LLM returned {resp.status_code} — waiting for recovery")
                    if cls._wait_for_server(60):
                        Log.info(f"Retrying after recovery (attempt {attempt})")
                        continue
                    last_error = f"HTTP {resp.status_code} and server did not recover"
                    break
                resp.raise_for_status()
                return cls._parse_stream(resp, stream_callback)
            except requests.ConnectionError as e:
                last_error = str(e)
                Log.warning(f"Connection lost (attempt {attempt}): {e}")
                if cls._wait_for_server(60) and attempt < Config.LLM_MAX_RETRIES:
                    continue
                break
            except requests.Timeout as e:
                last_error = str(e)
                Log.warning(f"Request timed out (attempt {attempt}): {e}")
                if attempt < Config.LLM_MAX_RETRIES:
                    time.sleep(Config.LLM_RETRY_DELAY * attempt)
            except requests.RequestException as e:
                last_error = str(e)
                if attempt < Config.LLM_MAX_RETRIES:
                    Log.warning(f"LLM error (attempt {attempt}): {e}")
                    time.sleep(Config.LLM_RETRY_DELAY * attempt)
        return {"error": f"LLM failed after {Config.LLM_MAX_RETRIES} attempts: {last_error}"}

    @staticmethod
    def _parse_stream(resp, stream_callback: Optional[Callable[[str], None]]) -> Dict[str, Any]:
        content       = ""
        tool_calls: Dict[int, Dict[str, Any]] = {}
        next_idx      = 0
        finish_reason = None
        in_think      = False

        resp.encoding = "utf-8"

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "): continue
            data = line[6:].strip()
            if data == "[DONE]": break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices: continue
            choice = choices[0]
            delta  = choice.get("delta", {})

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            raw_content = delta.get("content")
            if raw_content:
                content += raw_content
                if stream_callback:
                    for part in re.split(r'(</?think>)', raw_content, flags=re.IGNORECASE):
                        if not part: continue
                        if part.lower() == "<think>":
                            in_think = True
                        elif part.lower() == "</think>":
                            in_think = False
                        elif in_think:
                            sys.stdout.write(colored(part, Colors.GRAY))
                            sys.stdout.flush()
                        else:
                            stream_callback(part)

            thinking_blocks = delta.get("thinking")
            if isinstance(thinking_blocks, list) and stream_callback:
                for tb in thinking_blocks:
                    if tb.get("thinking"):
                        sys.stdout.write(colored(tb["thinking"], Colors.GRAY))
                        sys.stdout.flush()

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", next_idx)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id":       tc.get("id", f"tc_{idx}"),
                        "type":     "function",
                        "function": {"name": None, "arguments": ""},
                    }
                    next_idx += 1
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tool_calls[idx]["function"]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    tool_calls[idx]["function"]["arguments"] += fn["arguments"]

        if content and stream_callback:
            stream_callback("\n")

        calls, incomplete = [], False
        for i in sorted(tool_calls):
            tc      = tool_calls[i]
            fn_name = tc["function"]["name"]

            if not fn_name:
                Log.warning(f"Tool call {i} missing name — skipping")
                incomplete = True
                continue

            args_str = tc["function"]["arguments"]
            Log.info(f"[PARSE] '{fn_name}' args length={len(args_str)} "
                     f"first100={repr(args_str[:100])}")

            is_empty = not args_str or not args_str.strip()

            if is_empty or args_str.strip() == "{}":
                if fn_name in _REQUIRED_ARG_TOOLS:
                    Log.error(f"'{fn_name}' received empty/bare args — "
                              f"finish_reason={finish_reason!r}")
                    incomplete = True
                    tc["function"]["arguments"] = "{}"
                    tc["_truncated"] = True
                    calls.append(tc)
                    continue
                tc["function"]["arguments"] = "{}"
                calls.append(tc)
                continue

            try:
                json.loads(args_str)
                calls.append(tc)
            except json.JSONDecodeError as parse_err:
                Log.error(f"[PARSE] '{fn_name}' JSON decode failed at pos "
                          f"{parse_err.pos}: {parse_err.msg} — "
                          f"tail={repr(args_str[-80:])}")
                repaired = False
                if len(args_str) < 500:
                    opens, closes = args_str.count("{"), args_str.count("}")
                    if opens > closes:
                        candidate = args_str + "}" * (opens - closes)
                        try:
                            json.loads(candidate)
                            tc["function"]["arguments"] = candidate
                            Log.info(f"[PARSE] Auto-repaired '{fn_name}': "
                                     f"added {opens - closes} brace(s)")
                            incomplete = True
                            calls.append(tc)
                            repaired = True
                        except json.JSONDecodeError:
                            pass
                if not repaired:
                    Log.error(f"[PARSE] '{fn_name}' unrecoverable JSON — marking truncated "
                              f"({'short' if len(args_str) < 500 else 'large'} payload, "
                              f"{len(args_str)} chars)")
                    incomplete = True
                    tc["function"]["arguments"] = "{}"
                    tc["_truncated"] = True
                    calls.append(tc)

        if finish_reason == "length":
            Log.warning("⚠️  Generation stopped: output token limit hit (finish_reason=length)")
            incomplete = True
        elif finish_reason == "stop" and incomplete:
            Log.warning("⚠️  finish_reason=stop but tool calls were incomplete")
        elif finish_reason is None and incomplete:
            Log.warning("⚠️  finish_reason=None — stream may have ended prematurely")

        clean_content, _ = strip_thinking(content)
        return {
            "content":       clean_content,
            "tool_calls":    calls or None,
            "incomplete":    incomplete,
            "finish_reason": finish_reason,
        }


# =============================================================================
# COMPLETION DETECTION
# =============================================================================

_SKIP_PREFIXES = ("#", "//", "/*", "*", "--", "<!--", "'")
_SKIP_KEYWORDS = ("print(", "return ", "def ", "class ")
_ASKING_PHRASES = ("would you like", "what would you", "should i")


def detect_completion(content: str, has_tool_calls: bool) -> Tuple[bool, str]:
    if "TASK_COMPLETE" in content.upper() and not has_tool_calls:
        for line in content.split("\n"):
            if "TASK_COMPLETE" not in line.upper(): continue
            stripped = line.strip()
            if any(stripped.startswith(p) for p in _SKIP_PREFIXES): continue
            if any(k in stripped.lower() for k in _SKIP_KEYWORDS): continue
            return True, "Explicit TASK_COMPLETE"
    if not has_tool_calls and len(content.strip()) > 50:
        if any(q in content.lower() for q in _ASKING_PHRASES):
            return False, "Asking for input"
        return True, "Answer without tools"
    return False, "Not complete"


# =============================================================================
# PERMISSION SYSTEM
# =============================================================================

def should_ask_permission(tool_name: str, mode: PermissionMode) -> bool:
    if mode == PermissionMode.AUTO:   return False
    if mode == PermissionMode.MANUAL: return True
    return tool_name in Config.DESTRUCTIVE_TOOLS


def ask_permission(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[PermissionMode]]:
    sep = colored("─" * 60, Colors.YELLOW)
    print(f"\n{sep}")
    print(colored(f"  ⚡ Permission needed: {tool_name}", Colors.YELLOW, bold=True))
    print(sep)
    print(json.dumps(args, indent=2))
    print(sep)
    response = input(colored("\n  Allow? [y/n/auto]: ", Colors.YELLOW)).lower().strip()
    if response == "auto":
        print(colored("  → Auto-approve mode enabled for this session", Colors.GREEN))
        return True, PermissionMode.AUTO
    return (True, None) if response in ("y", "yes") else (False, None)


# =============================================================================
# SUB-AGENT EXECUTION
# shell + vision intentionally excluded — sub-agents are file-only.
# =============================================================================

def run_sub_agent(
    messages: List[Dict[str, Any]],
    workspace: Path,
    session_id: str,
    max_iterations: int,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    # Sub-agents are file-only: no shell, no vision, no git.
    # get_available_tools() is NOT used here intentionally — the allowlist controls scope.
    available_tools = [
        t for t in TOOL_SCHEMAS
        if t["function"]["name"] in _SUB_AGENT_TOOL_NAMES
    ]
    detector = LoopDetector()

    parent_ctx   = get_current_context()
    sub_todo_mgr = TodoManager(workspace, session_id)
    sub_plan_mgr = PlanManager(workspace, session_id)
    sub_task_mgr = TaskStateManager(workspace, session_id)
    set_tool_context(
        workspace, session_id,
        sub_todo_mgr, sub_plan_mgr, sub_task_mgr,
        messages, mode=parent_ctx.get("mode", "interactive"),
        stream_callback=stream_callback,
    )

    try:
        for iteration in range(1, max_iterations + 1):
            loop_msg = detector.check(iteration)
            if loop_msg:
                return {"status": "error", "output": f"Sub-agent stalled: {loop_msg}",
                        "iterations": iteration}

            response = LLMClient.call(messages, available_tools,
                                      stream_callback=stream_callback)
            if "error" in response:
                return {"status": "error", "output": response["error"],
                        "iterations": iteration}

            content    = response.get("content", "")
            tool_calls = response.get("tool_calls") or []

            if not content and not tool_calls:
                detector.track_empty()
                continue

            if detect_completion(content, bool(tool_calls))[0]:
                return {"status": "completed", "output": "File created",
                        "iterations": iteration}

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content:    assistant_msg["content"]    = content
            if tool_calls: assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

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

                handler = TOOL_HANDLERS.get(fn_name)
                try:
                    result  = (handler(workspace, **args) if handler
                               else {"success": False, "error": f"Unknown tool: {fn_name}"})
                    success = result.get("success", False)
                    detector.track_tool(fn_name, args, success)
                    if success: detector.track_success(iteration)
                    else:       detector.track_error()
                except Exception as e:
                    result = {"success": False, "error": str(e)}
                    detector.track_error()
                    detector.track_tool(fn_name, args, False)

                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                    "content": json.dumps(result),
                })

        return {"status": "max_iterations",
                "output": "Sub-agent reached max iterations",
                "iterations": max_iterations}

    finally:
        set_current_context(parent_ctx)


# =============================================================================
# PLAN MODE EXECUTION
# =============================================================================

def run_plan_mode(task: str, workspace: Path) -> Optional[Dict[str, Any]]:
    Log.plan("Building execution plan…")
    messages = [
        {"role": "system", "content": PLAN_MODE_PROMPT},
        {"role": "user",   "content": f"Create plan for:\n\n{task}"},
    ]
    # Planning uses a restricted read-only set — vision not needed here.
    planning_tools = [
        t for t in TOOL_SCHEMAS
        if t["function"]["name"] in _PLAN_TOOL_NAMES
    ]

    for iteration in range(1, 11):
        response = LLMClient.call(messages, planning_tools)
        if "error" in response:
            Log.error(f"Planning failed: {response['error']}")
            return None

        content    = response.get("content", "")
        tool_calls = response.get("tool_calls") or []

        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if content:    assistant_msg["content"]    = content
        if tool_calls: assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if "PLAN_APPROVED" in content:
            for msg in reversed(messages):
                if msg.get("role") != "assistant": continue
                body = msg.get("content", "")
                if "steps" not in body or "{" not in body: continue
                try:
                    start     = body.find("{")
                    end       = body.rfind("}") + 1
                    plan_data = json.loads(body[start:end])
                    for step in plan_data.get("steps", []):
                        step.setdefault("status", "pending")
                    return plan_data
                except json.JSONDecodeError:
                    pass

        for tc in tool_calls:
            fn_name, args_raw, tc_id = _unpack_tc(tc, f"tc_p{iteration}")
            args, err = _parse_tool_args(fn_name, args_raw)
            if err:
                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                    "content": json.dumps({"success": False, "error": err}),
                })
                continue
            handler = TOOL_HANDLERS.get(fn_name)
            try:
                result = (handler(workspace, **args) if handler
                          else {"success": False, "error": "Not available in plan mode"})
            except Exception as e:
                result = {"success": False, "error": str(e)}
            messages.append({
                "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                "content": json.dumps(result),
            })

    Log.warning("Plan mode: max iterations reached without PLAN_APPROVED")
    return None


# =============================================================================
# TOOL EXECUTION HELPERS
# =============================================================================

def _execute_tool(
    fn_name: str,
    args: Dict[str, Any],
    workspace: Path,
    available_tools: List[Dict],
    detector: LoopDetector,
    iteration: int,
    mcp_manager: Any,
    emit: Callable,
) -> Dict[str, Any]:
    schema = next((t for t in available_tools if t["function"]["name"] == fn_name), None)
    if schema:
        missing = [p for p in schema["function"]["parameters"].get("required", [])
                   if p not in args]
        if missing:
            detector.track_error()
            return {"success": False,
                    "error": f"Missing required parameters: {missing}. Got: {list(args.keys())}"}

    if len(json.dumps(args)) < 5:
        emit("warning", {"message": f"Very short args for '{fn_name}': {args}"})

    try:
        if fn_name.startswith("mcp_"):
            mcp_result = mcp_manager.call_tool(fn_name, args)
            if not mcp_result.get("success"):
                detector.track_error()
                detector.track_tool(fn_name, args, False)
                return {"success": False, "error": mcp_result.get("error", "MCP call failed")}
            blocks = mcp_result.get("result", {}).get("content", [])
            if isinstance(blocks, list) and blocks:
                text_parts = [b["text"] for b in blocks
                              if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
                if text_parts:
                    detector.track_success(iteration)
                    detector.track_tool(fn_name, args, True)
                    return {"success": True, "output": "\n\n".join(text_parts)}
                detector.track_error()
                detector.track_tool(fn_name, args, False)
                return {"success": False,
                        "error": f"MCP returned {len(blocks)} block(s) with no text"}
            elif isinstance(blocks, list):
                detector.track_error()
                detector.track_tool(fn_name, args, False)
                return {"success": False, "error": "MCP returned empty content"}
            else:
                detector.track_success(iteration)
                detector.track_tool(fn_name, args, True)
                return {"success": True, "output": str(mcp_result.get("result", {})),
                        "warning": "Non-standard MCP response format"}

        handler = TOOL_HANDLERS.get(fn_name)
        if not handler:
            detector.track_error()
            detector.track_tool(fn_name, args, False)
            return {"success": False, "error": f"Unknown tool: {fn_name}"}

        result  = handler(workspace, **args)
        success = result.get("success", False)
        detector.track_tool(fn_name, args, success)
        if success: detector.track_success(iteration)
        else:       detector.track_error()
        return result

    except Exception as e:
        emit("error", {"message": f"Tool exception in '{fn_name}': {e}"})
        detector.track_error()
        detector.track_tool(fn_name, args, False)
        return {"success": False, "error": str(e)}


def _process_tool_calls(
    tool_calls: List[Dict],
    workspace: Path,
    available_tools: List[Dict],
    detector: LoopDetector,
    iteration: int,
    mcp_manager: Any,
    messages: List[Dict],
    current_permission_mode: PermissionMode,
    emit: Callable,
) -> PermissionMode:
    """Execute all tool calls for one agent iteration.

    Returns the (possibly updated) permission mode.

    NOTE: returns a single value (PermissionMode), not a tuple.
    Old callers that did:
        _, mode = _process_tool_calls(...)
    should be updated to:
        mode = _process_tool_calls(...)
    """
    todo_op_count = 0
    total_calls   = len(tool_calls)

    for tc in tool_calls:
        fn_name, args_raw, tc_id = _unpack_tc(tc, f"tc_{iteration}")
        args, err = _parse_tool_args(fn_name, args_raw)

        if err:
            emit("error", {"message": err})
            detector.track_error()
            messages.append({
                "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                "content": json.dumps({"success": False, "error": err}),
            })
            continue

        if should_ask_permission(fn_name, current_permission_mode):
            allowed, new_mode = ask_permission(fn_name, args)
            if not allowed:
                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                    "content": json.dumps({"success": False, "error": "Permission denied"}),
                })
                continue
            if new_mode:
                current_permission_mode = new_mode

        emit("tool_call", {"name": fn_name, "args_preview": json.dumps(args)[:80]})
        result = _execute_tool(fn_name, args, workspace, available_tools,
                               detector, iteration, mcp_manager, emit)

        if fn_name.startswith("todo_"):
            todo_op_count += 1

        if result.get("success"):
            emit("tool_result", {"name": fn_name, "success": True,
                                  "summary": str(result)[:200]})
            if fn_name == "todo_complete" and result.get("all_complete"):
                emit("log", {"message": "✓ All todos complete"})
                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                    "content": json.dumps(result),
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "✅ ALL TODOS COMPLETE.\n\n"
                        "Your task is finished. Output TASK_COMPLETE right now.\n"
                        "Do NOT call any more tools."
                    ),
                })
                continue
        else:
            emit("tool_result", {"name": fn_name, "success": False,
                                  "error": result.get("error", "")[:200]})
            if (fn_name == "todo_complete"
                    and result.get("already_completed")
                    and result.get("all_complete")):
                emit("warning", {"message": "Loop: completing already-done todo"})
                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "name": fn_name,
                    "content": json.dumps(result),
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ LOOP DETECTED: You are re-completing finished todos.\n\n"
                        "ALL TODOS ARE DONE. Output TASK_COMPLETE now."
                    ),
                })
                continue

        result_str = json.dumps(result)
        if len(result_str) > Config.MAX_TOOL_OUTPUT:
            result_str = truncate_output(result_str, Config.MAX_TOOL_OUTPUT, fn_name)
            result = {"success": result.get("success", False),
                      "truncated": True, "data": result_str}

        messages.append({
            "role": "tool", "tool_call_id": tc_id, "name": fn_name,
            "content": json.dumps(result),
        })

    if todo_op_count > 0 and todo_op_count == total_calls:
        todo_mgr = get_current_context().get("todo_manager")
        if todo_mgr:
            still_pending = [t for t in todo_mgr.list_all().get("todos", [])
                             if t["status"] not in ("completed", "blocked")]
            if not still_pending:
                emit("warning", {"message": "Todo-only iteration, nothing pending → hard stop"})
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ HARD STOP: You've been doing todo bookkeeping with nothing left to do.\n\n"
                        "If your deliverable is verified → output TASK_COMPLETE\n"
                        "If blocked → output BLOCKED: <reason>\n"
                        "No more todo calls."
                    ),
                })

    return current_permission_mode


# =============================================================================
# STREAMING HEADER HELPER
# =============================================================================

class _HeaderStreamCb:
    def __init__(self, inner_cb: Optional[Callable[[str], None]], mode: str):
        self._inner   = inner_cb
        self._mode    = mode
        self._printed = False

    def reset(self) -> None:
        self._printed = False

    @property
    def printed(self) -> bool:
        return self._printed

    def __call__(self, token: str) -> None:
        if not self._printed and self._mode == "interactive":
            self._printed = True
            sys.stdout.write(colored("\nAssistant:\n\n", Colors.CYAN, bold=True))
            sys.stdout.flush()
        if self._inner:
            self._inner(token)
