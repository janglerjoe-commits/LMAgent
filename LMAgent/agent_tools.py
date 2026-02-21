#!/usr/bin/env python3
"""
agent_tools.py — Tool layer for LMAgent.

Sandbox patch v9.5.0-sandbox:
  - Re-introduces the shell tool using a cross-platform sandboxed subprocess.
  - Windows + pywin32  : Windows Job Object (kill_on_job_close, memory limit).
  - Windows, no pywin32: psutil process-tree kill on timeout.
  - macOS / Linux      : os.setsid() process group + RLIMIT_AS + RLIMIT_CPU.
  - All paths still validated through Safety.validate_command() before execution.
  - sandboxed_shell.py must live in the same directory as this file.
  - pip install psutil        # all platforms
  - pip install pywin32       # Windows only — stronger Job Object backend

All other behaviour is unchanged from v9.4.0-nosell.
"""

import json
import os
import re
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from agent_core import (
    Config, Colors, Log, PermissionMode, _IS_WINDOWS,
    truncate_output, strip_thinking,
    get_current_context, set_current_context, _get_ctx,
    Safety, FileEditor,
    get_shell_session,
    TodoManager, PlanManager, TaskStateManager, TaskState,
    SessionManager, LoopDetector,
    colored,
)
from sandboxed_shell import run_sandboxed, sandbox_info

# Backward-compat alias for any external code that imported get_powershell_session.
get_powershell_session = get_shell_session


# =============================================================================
# SHELL QUOTING UTILITIES  (used only by git helpers — not exposed to LLM)
# =============================================================================

def _shell_quote(s: str) -> str:
    if _IS_WINDOWS:
        escaped = s.replace("`", "``").replace('"', '`"').replace("$", "`$")
        return f'"{escaped}"'
    return shlex.quote(s)


_GIT_REF_RE = re.compile(r'^[A-Za-z0-9_.\-/]+$')


def _validate_git_ref(name: str) -> Tuple[bool, str]:
    if not name:
        return False, "Empty ref name"
    if not _GIT_REF_RE.match(name):
        bad = sorted({c for c in name if not re.match(r'[A-Za-z0-9_.\-/]', c)})
        return False, (
            f"Ref name '{name}' contains disallowed characters: {bad!r}. "
            "Only alphanumerics, hyphens, underscores, dots, and slashes are permitted."
        )
    if ".." in name:
        return False, f"Ref name '{name}' contains '..' (path-traversal sequence)"
    return True, ""


# =============================================================================
# SECURE GIT EXECUTION HELPER  (internal only — not a tool the LLM can call)
# =============================================================================

def _git_safe(workspace: Path, cmd: str) -> Tuple[str, int]:
    ok, reason = Safety.validate_command(cmd, workspace)
    if not ok:
        raise ValueError(reason)
    return get_shell_session(workspace).execute(cmd)


# =============================================================================
# SHARED TOOL-CALL HELPERS
# =============================================================================

def _unpack_tc(tc: Dict[str, Any], fallback_id: str) -> Tuple[str, str, str]:
    fn = tc.get("function") or {}
    return (
        fn.get("name", "").strip(),
        fn.get("arguments", ""),
        tc.get("id") or fallback_id,
    )


def _parse_tool_args(fn_name: str, args_raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not args_raw or not args_raw.strip():
        return {}, None
    try:
        return json.loads(args_raw), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in tool arguments for '{fn_name}': {e}"


# =============================================================================
# TOOL HANDLERS — FILE OPERATIONS
# =============================================================================

def tool_get_time(workspace: Path) -> Dict[str, Any]:
    now = datetime.now()
    return {
        "success":   True,
        "datetime":  now.isoformat(),
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "weekday":   now.strftime("%A"),
        "timestamp": int(now.timestamp()),
    }


def tool_read(workspace: Path, path: str) -> Dict[str, Any]:
    ok, err, fp = Safety.validate_path(workspace, path, must_exist=True)
    if not ok:
        return {"success": False, "error": err}
    if not fp.is_file():
        return {"success": False, "error": "Not a file"}
    if fp.suffix.lower() in Config.BINARY_EXTS:
        return {"success": False, "error": "Cannot read binary file"}
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
        return {"success": True,
                "content": truncate_output(content, Config.MAX_FILE_READ, path),
                "lines":   content.count("\n") + 1}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_write(workspace: Path, path: str, content: str) -> Dict[str, Any]:
    ok, err, fp = Safety.validate_path(workspace, path)
    if not ok:
        return {"success": False, "error": err}
    if fp.exists() and fp.suffix.lower() in Config.BINARY_EXTS:
        return {"success": False, "error": "Cannot overwrite binary file"}
    try:
        existed = fp.exists()
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(str(tmp), str(fp))
        return {"success": True,
                "path":    str(fp.relative_to(workspace)).replace("\\", "/"),
                "action":  "modified" if existed else "created",
                "size":    len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_edit(workspace: Path, path: str, search: str, replace: str) -> Dict[str, Any]:
    ok, err, fp = Safety.validate_path(workspace, path, must_exist=True)
    if not ok:
        return {"success": False, "error": err}
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
        success, new_content, msg = FileEditor.search_replace(content, search, replace)
        if not success:
            return {"success": False, "error": truncate_output(msg, 500)}
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(str(tmp), str(fp))
        return {"success": True,
                "path":    str(fp.relative_to(workspace)).replace("\\", "/"),
                "method":  msg}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_glob(workspace: Path, pattern: str) -> Dict[str, Any]:
    try:
        matches = list(workspace.glob(pattern))
        files   = []
        for m in matches[:Config.MAX_LS_ENTRIES]:
            if m.is_file() and not any(p.startswith(".") for p in m.parts):
                try:
                    files.append(str(m.relative_to(workspace)).replace("\\", "/"))
                except Exception:
                    files.append(str(m).replace("\\", "/"))
        return {"success": True, "files": files, "count": len(files),
                "total_matches": len(matches),
                "truncated":     len(matches) > Config.MAX_LS_ENTRIES}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_grep(workspace: Path, pattern: str,
              paths: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        if paths:
            search_files: List[Path] = []
            for p in paths:
                if "*" in p:
                    search_files.extend(workspace.glob(p))
                else:
                    fp = workspace / p
                    if fp.exists():
                        search_files.append(fp)
        else:
            search_files = [f for f in workspace.rglob("*")
                            if f.is_file()
                            and not any(p.startswith(".") for p in f.parts)]
        matches: List[Dict[str, Any]] = []
        for fp in search_files[:100]:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if pattern in line:
                            matches.append({
                                "file":    str(fp.relative_to(workspace)).replace("\\", "/"),
                                "line":    i,
                                "content": line.rstrip()[:200],
                            })
                        if len(matches) >= Config.MAX_GREP_RESULTS:
                            break
                if len(matches) >= Config.MAX_GREP_RESULTS:
                    break
            except Exception:
                continue
        return {"success":   True,
                "matches":   matches,
                "truncated": len(matches) >= Config.MAX_GREP_RESULTS}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_ls(workspace: Path, path: str = ".") -> Dict[str, Any]:
    ok, err, dp = Safety.validate_path(workspace, path)
    if not ok:
        return {"success": False, "error": err}
    if not dp.exists():
        return {"success": False, "error": "Path does not exist"}
    if not dp.is_dir():
        return {"success": False, "error": "Not a directory"}
    try:
        all_entries = list(dp.iterdir())
        entries     = []
        for item in sorted(all_entries)[:Config.MAX_LS_ENTRIES]:
            if item.name.startswith("."):
                continue
            stat = item.stat()
            entries.append({"name": item.name,
                             "type": "dir" if item.is_dir() else "file",
                             "size": stat.st_size if item.is_file() else 0})
        return {"success": True, "entries": entries,
                "total":     len(all_entries),
                "truncated": len(all_entries) > Config.MAX_LS_ENTRIES}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_mkdir(workspace: Path, path: str) -> Dict[str, Any]:
    ok, err, dp = Safety.validate_path(workspace, path)
    if not ok:
        return {"success": False, "error": err}
    try:
        if dp.exists():
            return {"success": True, "message": "Already exists"}
        dp.mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": "Created"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL HANDLERS — SANDBOXED SHELL
# =============================================================================

def tool_shell(
    workspace: Path,
    command: str,
    timeout: int = 30,
    max_memory_mb: int = 512,
) -> Dict[str, Any]:
    """
    Run *command* inside a platform-appropriate sandbox.

    Sandbox backends (auto-selected):
      • Windows + pywin32  : Job Object — kill_on_job_close + memory limit.
        The process tree is hard-killed the moment the handle closes, even if
        Python itself crashes.
      • Windows, no pywin32: psutil tree-kill on timeout.
      • macOS / Linux      : new process group (os.setsid) + RLIMIT_AS +
        RLIMIT_CPU — entire group killed on timeout or error.

    The command is validated by Safety.validate_command() before execution.
    stdout and stderr are merged; output is capped at 32 KB.

    Install:
        pip install psutil          # all platforms
        pip install pywin32         # Windows only, for stronger Job Object backend
    """
    # ── safety gate (same check used by _git_safe) ───────────────────────────
    ok, reason = Safety.validate_command(command, workspace)
    if not ok:
        return {"success": False, "error": f"Command blocked by safety check: {reason}"}

    # ── clamp user-supplied limits to sane bounds ─────────────────────────────
    timeout       = max(1, min(timeout, 120))           # 1 – 120 s
    max_memory_mb = max(64, min(max_memory_mb, 2048))   # 64 MB – 2 GB

    try:
        output, exit_code = run_sandboxed(
            cmd=command,
            workspace=workspace,
            timeout=timeout,
            max_output_bytes=32_768,
            max_memory_mb=max_memory_mb,
        )
        success = exit_code == 0
        return {
            "success":   success,
            "exit_code": exit_code,
            "output":    output,
            "sandbox":   sandbox_info()["backend"],
        }
    except Exception as e:
        return {"success": False, "error": f"Sandbox error: {e}"}


# =============================================================================
# TOOL HANDLERS — GIT  (use _git_safe internally; never exposed as raw shell)
# =============================================================================

def tool_git_status(workspace: Path) -> Dict[str, Any]:
    try:
        output, code = _git_safe(workspace, "git status --porcelain")
        if code != 0:
            return {"success": False, "error": "Not a git repository"}
        files: Dict[str, List[str]] = {
            "modified": [], "added": [], "deleted": [], "untracked": []
        }
        for line in output.splitlines()[:Config.MAX_LS_ENTRIES]:
            if len(line) < 3:
                continue
            status, path = line[:2], line[3:]
            if   "M" in status: files["modified"].append(path)
            elif "A" in status: files["added"].append(path)
            elif "D" in status: files["deleted"].append(path)
            elif "?" in status: files["untracked"].append(path)
        branch, _ = _git_safe(workspace, "git branch --show-current")
        total      = sum(len(v) for v in files.values())
        return {"success": True, "branch": branch.strip(), **files,
                "total_changes": total,
                "truncated":     total > Config.MAX_LS_ENTRIES}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_diff(workspace: Path, path: str = "", staged: bool = False) -> Dict[str, Any]:
    try:
        if path:
            ok, err, resolved = Safety.validate_path(workspace, path)
            if not ok:
                return {"success": False, "error": f"Invalid diff path: {err}"}
        cmd = "git diff"
        if staged:
            cmd += " --staged"
        if path:
            cmd += f" -- {_shell_quote(path)}"
        output, code = _git_safe(workspace, cmd)
        if code != 0:
            return {"success": False, "error": "Git diff failed"}
        return {"success":     True,
                "diff":        truncate_output(output, Config.MAX_TOOL_OUTPUT),
                "has_changes": bool(output.strip())}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_add(workspace: Path, paths: List[str]) -> Dict[str, Any]:
    try:
        quoted: List[str] = []
        for p in paths:
            ok, err, _ = Safety.validate_path(workspace, p)
            if not ok:
                return {"success": False, "error": f"{p}: {err}"}
            quoted.append(_shell_quote(p))
        cmd = f"git add {' '.join(quoted)}"
        output, code = _git_safe(workspace, cmd)
        if code != 0:
            return {"success": False, "error": truncate_output(output, 500)}
        return {"success": True, "staged": len(paths)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_commit(workspace: Path, message: str, allow_empty: bool = False) -> Dict[str, Any]:
    try:
        cmd = f"git commit -m {_shell_quote(message)}"
        if allow_empty:
            cmd += " --allow-empty"
        output, code = _git_safe(workspace, cmd)
        if code != 0:
            return {"success": False, "error": truncate_output(output, 500)}
        rev, _ = _git_safe(workspace, "git rev-parse HEAD")
        return {"success": True, "commit": rev.strip()[:8]}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_git_branch(workspace: Path, name: str = "",
                    create: bool = False, switch: bool = False) -> Dict[str, Any]:
    try:
        if not name:
            output, code = _git_safe(workspace, "git branch")
            if code != 0:
                return {"success": False, "error": "Failed to list branches"}
            branches, current = [], None
            for line in output.splitlines()[:20]:
                line = line.strip()
                if line.startswith("* "):
                    current = line[2:]
                elif line:
                    branches.append(line)
            return {"success": True, "action": "list",
                    "branches": branches, "current": current}
        ok, err = _validate_git_ref(name)
        if not ok:
            return {"success": False, "error": err}
        git_cmd = ("git checkout -b" if create else
                   "git checkout"    if switch  else "git branch")
        action  = "created" if create else ("switched" if switch else "created_local")
        output, code = _git_safe(workspace, f"{git_cmd} {_shell_quote(name)}")
        if code != 0:
            return {"success": False, "error": truncate_output(output, 500)}
        return {"success": True, "action": action, "branch": name}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL HANDLERS — TODO & PLAN
# =============================================================================

def set_tool_context(
    workspace: Path,
    session_id: str,
    todo_manager: TodoManager,
    plan_manager: PlanManager,
    task_state_manager: TaskStateManager,
    messages: Optional[List[Dict[str, Any]]] = None,
    mode: str = "interactive",
    stream_callback: Optional[Callable] = None,
):
    set_current_context({
        "workspace":          workspace,
        "session_id":         session_id,
        "todo_manager":       todo_manager,
        "plan_manager":       plan_manager,
        "task_state_manager": task_state_manager,
        "messages":           messages,
        "mode":               mode,
        "stream_callback":    stream_callback,
    })


def tool_todo_add(workspace: Path, description: str, notes: str = "") -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    return mgr.add(description, notes)


def tool_todo_complete(workspace: Path, todo_id: int) -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    existing = next((t for t in mgr.todos if t.id == todo_id), None)
    if existing and existing.status == "completed":
        summary  = mgr.list_all()
        total    = summary.get("total", 0)
        done     = summary.get("completed", 0)
        all_done = total > 0 and done >= total
        return {
            "success":           False,
            "error":             f"Todo #{todo_id} is already completed.",
            "already_completed": True,
            "all_complete":      all_done,
            "completed_count":   done,
            "total_count":       total,
            "message": ("ALL TODOS DONE. Output TASK_COMPLETE now."
                        if all_done else f"{done}/{total} todos complete."),
        }
    result = mgr.complete(todo_id)
    if result.get("success"):
        summary  = mgr.list_all()
        total    = summary.get("total", 0)
        done     = summary.get("completed", 0)
        all_done = total > 0 and done >= total
        result.update({
            "all_complete":    all_done,
            "completed_count": done,
            "total_count":     total,
            "message": (
                f"Todo #{todo_id} complete. ALL {total} TODOS DONE. "
                "YOUR TASK IS FINISHED. Output TASK_COMPLETE now. Do not call any more tools."
                if all_done else f"Todo #{todo_id} complete. {done}/{total} done."
            ),
        })
    return result


def tool_todo_update(workspace: Path, todo_id: int,
                     status: str, notes: str = "") -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    valid = {"pending", "in_progress", "completed", "blocked"}
    if status not in valid:
        return {"success": False,
                "error": f"Invalid status '{status}'. Must be one of: {sorted(valid)}"}
    return mgr.update_status(todo_id, status, notes)


def tool_todo_list(workspace: Path) -> Dict[str, Any]:
    mgr, err = _get_ctx("todo_manager", "Todo manager")
    if err: return err
    result = mgr.list_all()
    if result.get("success"):
        total    = result.get("total", 0)
        done     = result.get("completed", 0)
        all_done = total > 0 and done >= total
        result["all_complete"] = all_done
        if all_done:
            result["hint"] = ("ALL TODOS COMPLETE. If your task deliverable is verified, "
                              "output TASK_COMPLETE now.")
    return result


def tool_plan_complete_step(workspace: Path, step_id: str,
                             verification_notes: str = "") -> Dict[str, Any]:
    mgr, err = _get_ctx("plan_manager", "Plan manager")
    if err: return err
    if not mgr.plan:
        return {"success": False, "error": "No active plan"}
    step = next((s for s in mgr.plan["steps"] if s["id"] == step_id), None)
    if not step:
        return {"success": False, "error": f"Step '{step_id}' not found"}
    if step["status"] == "completed":
        return {"success": False, "error": f"Step '{step_id}' already completed"}
    if step["status"] == "pending":
        return {"success": False, "error": f"Step '{step_id}' not started yet"}
    mgr.complete_step(step_id)
    return {"success":            True,
            "step_id":            step_id,
            "description":        step.get("description", ""),
            "verification_notes": verification_notes,
            "message":            f"Step '{step_id}' marked complete"}


# =============================================================================
# TOOL HANDLERS — TASK STATE
# =============================================================================

def tool_task_state_update(
    workspace: Path,
    objective: str = "",
    completion_gate: str = "",
    total_count: int = -1,
    processed_count: int = -1,
    remaining_queue: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    last_error: str = "",
    recovery_instruction: str = "",
    next_action: str = "",
) -> Dict[str, Any]:
    mgr, err = _get_ctx("task_state_manager", "Task state manager")
    if err: return err
    state = mgr.current_state
    if not state:
        if not objective:
            return {"success": False, "error": "objective required for new task state"}
        state = TaskState(
            objective=objective,
            completion_gate=completion_gate or "processed == total",
            inventory_hash="",
            total_count=max(total_count, 0),
            processed_count=max(processed_count, 0),
            remaining_queue=remaining_queue or [],
            rename_map=rename_map or {},
            last_error=last_error,
            recovery_instruction=recovery_instruction,
            next_action=next_action,
            last_updated="",
        )
    else:
        if objective:                   state.objective            = objective
        if completion_gate:             state.completion_gate      = completion_gate
        if total_count     >= 0:        state.total_count          = total_count
        if processed_count >= 0:        state.processed_count      = processed_count
        if remaining_queue is not None: state.remaining_queue      = remaining_queue
        if rename_map      is not None: state.rename_map.update(rename_map)
        if last_error:                  state.last_error           = last_error
        if recovery_instruction:        state.recovery_instruction = recovery_instruction
        if next_action:                 state.next_action          = next_action
    if remaining_queue:
        state.inventory_hash = TaskState.compute_inventory_hash(remaining_queue)
    mgr.checkpoint(state)
    return {
        "success": True,
        "state":   state.to_dict(),
        "message": f"Checkpointed: {state.processed_count}/{state.total_count} processed",
    }


def tool_task_state_get(workspace: Path) -> Dict[str, Any]:
    mgr, err = _get_ctx("task_state_manager", "Task state manager")
    if err: return err
    return {"success": True,
            "state": mgr.current_state.to_dict() if mgr.current_state else None}


def tool_task_reconcile(workspace: Path) -> Dict[str, Any]:
    mgr, err = _get_ctx("task_state_manager", "Task state manager")
    if err: return err
    if not mgr.current_state:
        return {"success": False, "error": "No task state to reconcile"}
    s = mgr.current_state
    return {
        "success": True,
        "message": (
            f"Reconciliation checkpoint. Current: {s.processed_count}/{s.total_count} processed. "
            f"You MUST: 1) Re-enumerate targets via ls/glob, "
            f"2) Update remaining_queue, 3) Continue from next unprocessed file."
        ),
        "current_state":   s.to_dict(),
        "action_required": "re_enumerate",
    }


# =============================================================================
# SUB-AGENT CONTEXT HELPER
# =============================================================================

def _build_parent_context(parent_messages: List[Dict[str, Any]],
                           max_chars: int = 12000) -> str:
    sections: List[str] = []
    for msg in reversed(parent_messages):
        role = msg.get("role", "")
        if role == "tool":
            name = msg.get("name", "tool")
            raw  = msg.get("content", "")
            try:
                parsed = json.loads(raw)
                text   = (parsed.get("output") or parsed.get("stdout")
                          or parsed.get("content") or parsed.get("result") or raw)
            except (json.JSONDecodeError, AttributeError):
                text = raw
            text = str(text).strip()
            if text:
                sections.append(f"[{name} result]\n{text[:3000]}")
        elif role == "assistant":
            content = msg.get("content", "")
            if content and len(content) > 40:
                sections.append(f"[reasoning]\n{content[:800]}")
    if not sections:
        return ""
    sections.reverse()
    combined = "\n\n---\n\n".join(sections)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[...trimmed]"
    return combined


def tool_task(workspace: Path, task_type: str,
              instructions: str, file_path: str) -> Dict[str, Any]:
    if not Config.ENABLE_SUB_AGENTS:
        return {"success": False, "error": "Sub-agents disabled"}
    try:
        Log.task(f"Sub-agent: Create {task_type} → {file_path}")
        ctx             = get_current_context()
        parent_messages = ctx.get("messages") or []
        ctx_content     = _build_parent_context(parent_messages)
        if ctx_content:
            ctx_note = ("\n\nPARENT CONTEXT (research/data gathered by parent agent):"
                        f"\n\n{ctx_content}")
            step1    = "Use the parent context above as your primary data source"
        else:
            ctx_note, step1 = "", "Create the file at the specified path"
        sub_sid = SessionManager(ctx["workspace"]).create(
            f"[CREATE-{task_type}] {file_path}",
            parent_session=ctx["session_id"],
        )
        sub_messages = [
            {"role": "system", "content": SUB_AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"CREATE FILE: {file_path}\n"
                f"TYPE: {task_type}"
                f"{ctx_note}\n\n"
                f"INSTRUCTIONS:\n{instructions}\n\n"
                f"REQUIREMENTS:\n"
                f"1. {step1}\n"
                f"2. Create the file at the specified path\n"
                f"3. Follow the instructions exactly\n"
                f"4. Use the write tool to save it\n"
                f"5. Say TASK_COMPLETE when done\n\n"
                f"Do NOT ask questions. Create the file now."
            )},
        ]
        result = run_sub_agent(
            messages=sub_messages,
            workspace=ctx["workspace"],
            session_id=sub_sid,
            max_iterations=Config.MAX_SUB_AGENT_ITERATIONS,
            stream_callback=ctx.get("stream_callback"),
        )
        Log.task(f"Sub-agent finished: {result['status']}")
        return {"success":    result["status"] == "completed",
                "file_path":  file_path,
                "iterations": result["iterations"],
                "session_id": sub_sid}
    except Exception as e:
        Log.error(f"Sub-agent failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},

    {"type": "function", "function": {
        "name": "read", "description": "Read file contents (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Create or overwrite a file (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit file with fuzzy search/replace (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "search": {"type": "string"},
                                      "replace": {"type": "string"}},
                       "required": ["path", "search", "replace"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files matching a glob pattern (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search for a text pattern in files (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                      "paths": {"type": "array", "items": {"type": "string"}}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "ls", "description": "List directory contents (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "mkdir", "description": "Create a directory (workspace only).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},

    # ── sandboxed shell ───────────────────────────────────────────────────────
    {"type": "function", "function": {
        "name": "shell",
        "description": (
            "Run a shell command inside a platform sandbox (bash on macOS/Linux, "
            "cmd on Windows). stdout and stderr are merged. Output is capped at 32 KB. "
            "Timeout defaults to 30 s (max 120 s). Memory defaults to 512 MB (max 2 GB). "
            "Use for build commands, tests, installers, package managers, or any "
            "operation the file tools cannot handle directly. "
            "All commands are validated by the safety layer before execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds before the process is killed (1–120, default 30).",
                },
                "max_memory_mb": {
                    "type": "integer",
                    "description": "Memory limit in MB (64–2048, default 512).",
                },
            },
            "required": ["command"],
        }}},

    {"type": "function", "function": {
        "name": "git_status", "description": "Get git status of the workspace.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "git_diff", "description": "Show git diff.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "staged": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "git_add", "description": "Stage files for commit.",
        "parameters": {"type": "object",
                       "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                       "required": ["paths"]}}},
    {"type": "function", "function": {
        "name": "git_commit", "description": "Commit staged changes.",
        "parameters": {"type": "object",
                       "properties": {"message": {"type": "string"},
                                      "allow_empty": {"type": "boolean"}},
                       "required": ["message"]}}},
    {"type": "function", "function": {
        "name": "git_branch",
        "description": "Manage git branches. No args → list. With name+flags → create/switch.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string"},
                                      "create": {"type": "boolean"},
                                      "switch": {"type": "boolean"}}}}},

    {"type": "function", "function": {
        "name": "todo_add",
        "description": "Add a todo item. WARNING: Only add todos BEFORE execution begins.",
        "parameters": {"type": "object",
                       "properties": {"description": {"type": "string"},
                                      "notes": {"type": "string"}},
                       "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "todo_complete",
        "description": "Mark a todo completed. Each todo_id must be completed ONCE. When all_complete=true, immediately output TASK_COMPLETE.",
        "parameters": {"type": "object",
                       "properties": {"todo_id": {"type": "integer"}},
                       "required": ["todo_id"]}}},
    {"type": "function", "function": {
        "name": "todo_update", "description": "Update todo status.",
        "parameters": {"type": "object",
                       "properties": {"todo_id": {"type": "integer"},
                                      "status": {"type": "string",
                                                 "enum": ["pending", "in_progress",
                                                          "completed", "blocked"]},
                                      "notes": {"type": "string"}},
                       "required": ["todo_id", "status"]}}},
    {"type": "function", "function": {
        "name": "todo_list",
        "description": "List todos. If all_complete=true is returned, output TASK_COMPLETE immediately.",
        "parameters": {"type": "object", "properties": {}}}},

    {"type": "function", "function": {
        "name": "plan_complete_step",
        "description": "Mark a plan step complete after verification.",
        "parameters": {"type": "object",
                       "properties": {"step_id": {"type": "string",
                                                   "description": "Step ID from the plan (e.g. 'step_1')"},
                                      "verification_notes": {"type": "string"}},
                       "required": ["step_id"]}}},

    {"type": "function", "function": {
        "name": "task",
        "description": "Delegate file creation to an isolated sub-agent.",
        "parameters": {"type": "object",
                       "properties": {"task_type": {"type": "string",
                                                     "description": "File type (html, css, js, python, etc.)"},
                                      "instructions": {"type": "string",
                                                       "description": "Full specification for the file"},
                                      "file_path": {"type": "string",
                                                    "description": "Where to save the file"}},
                       "required": ["task_type", "instructions", "file_path"]}}},

    {"type": "function", "function": {
        "name": "task_state_update",
        "description": "Checkpoint task progress. Call after each file processed.",
        "parameters": {"type": "object",
                       "properties": {
                           "objective":            {"type": "string"},
                           "completion_gate":      {"type": "string"},
                           "total_count":          {"type": "integer"},
                           "processed_count":      {"type": "integer"},
                           "remaining_queue":      {"type": "array", "items": {"type": "string"}},
                           "rename_map":           {"type": "object"},
                           "last_error":           {"type": "string"},
                           "recovery_instruction": {"type": "string"},
                           "next_action":          {"type": "string"},
                       }}}},
    {"type": "function", "function": {
        "name": "task_state_get",
        "description": "Retrieve current task state checkpoint.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_reconcile",
        "description": "Reconcile after rename/move operations. Re-enumerate from disk.",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOL_HANDLERS: Dict[str, Callable] = {
    "get_time":           tool_get_time,
    "shell":              tool_shell,
    "read":               tool_read,
    "write":              tool_write,
    "edit":               tool_edit,
    "glob":               tool_glob,
    "grep":               tool_grep,
    "ls":                 tool_ls,
    "mkdir":              tool_mkdir,
    "git_status":         tool_git_status,
    "git_diff":           tool_git_diff,
    "git_add":            tool_git_add,
    "git_commit":         tool_git_commit,
    "git_branch":         tool_git_branch,
    "todo_add":           tool_todo_add,
    "todo_complete":      tool_todo_complete,
    "todo_update":        tool_todo_update,
    "todo_list":          tool_todo_list,
    "plan_complete_step": tool_plan_complete_step,
    "task":               tool_task,
    "task_state_update":  tool_task_state_update,
    "task_state_get":     tool_task_state_get,
    "task_reconcile":     tool_task_reconcile,
}

_REQUIRED_ARG_TOOLS = frozenset({
    "shell",
    "write", "read", "edit", "glob", "grep", "ls", "mkdir",
    "todo_add", "todo_complete", "todo_update", "plan_complete_step",
    "git_add", "git_commit", "git_branch", "git_diff",
    "task",
})

_llm_local = threading.local()


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
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a skilled, proactive digital coworker. You have real tools that make real changes.

YEAR: 2026. Think modern. Don't mention the year.

{soul_section}

════════════════════════════════════════════════════
THE ONE RULE
════════════════════════════════════════════════════

Every single iteration: call a tool OR output TASK_COMPLETE. Nothing else is valid.

If you find yourself writing prose without a tool call and without TASK_COMPLETE,
you are broken. Stop. Output TASK_COMPLETE or call a tool.

════════════════════════════════════════════════════
TASK_COMPLETE — READ THIS CAREFULLY
════════════════════════════════════════════════════

TASK_COMPLETE means the task is done. Output it the moment the deliverable exists
and is verified. It is a hard stop — nothing follows it, ever.

Output TASK_COMPLETE immediately when ANY of these are true:
  • You verified the file/output exists and is correct
  • todo_complete returned all_complete: true
  • The user's question has been answered
  • The plan is finished and verified

DO NOT delay TASK_COMPLETE to:
  • "summarize what was done"
  • "let the user know it's done"
  • run one more verification
  • complete todos you already completed
  • explain your reasoning

TASK_COMPLETE. Stop. Done. That's it.

════════════════════════════════════════════════════
LOOP PREVENTION — MANDATORY SELF-CHECK
════════════════════════════════════════════════════

Before every tool call, ask yourself:
  1. Did this exact operation succeed already this session? → Don't repeat it.
  2. Have I verified the deliverable exists? → If yes, output TASK_COMPLETE.
  3. Am I calling todo_complete on an already-completed todo? → STOP. Output TASK_COMPLETE.
  4. Have I done 3+ iterations without meaningful new output? → Output TASK_COMPLETE or BLOCKED.

Signs you are in a loop — stop immediately:
  • Calling the same tool with the same args twice
  • Re-reading a file you already read with no changes made
  • Completing todos you already completed
  • Verifying something you already verified
  • Writing text about being done instead of outputting TASK_COMPLETE

════════════════════════════════════════════════════
ACT FIRST, NARRATE NEVER
════════════════════════════════════════════════════

  • 3-sentence reasoning cap — then act
  • Never narrate what you're about to do — just do it
  • One reasoning pass = one tool call. Always.
  • If you know the answer or the file is written — TASK_COMPLETE now.

════════════════════════════════════════════════════
TOOL CALL DISCIPLINE
════════════════════════════════════════════════════

Every call needs ALL required arguments. No exceptions.

  write  → "path" AND "content" (complete file, not a placeholder)
  read   → "path"
  edit   → "path", "search", "replace"
  shell  → "command" (timeout and max_memory_mb are optional)

When unsure of a path: ls or glob first, then act.

════════════════════════════════════════════════════
EXECUTION RULES
════════════════════════════════════════════════════

0. STOP THE MOMENT THE JOB IS DONE
   Deliverable verified → TASK_COMPLETE → stop. Zero exceptions.

1. VERIFY ONCE, THEN STOP
   After write/edit → verify with read or ls exactly once.
   Verification passed → TASK_COMPLETE. Do not verify again.
   Verification failed → fix it. One fix attempt, then TASK_COMPLETE or BLOCKED.

2. ALL MEANS ALL — BUT COUNT FIRST
   "all files" → enumerate with glob/ls, process each one, then TASK_COMPLETE.
   Do not re-enumerate files you've already processed.

3. EXECUTE, DON'T SCRIPT
   Writing a script is not completing a task. Make the changes directly.
   Exception: if the task IS to write a script, write it then TASK_COMPLETE.

4. TODOS — STRICT RULES
   • todo_complete on each ID exactly once — never twice
   • todo_complete returns all_complete: true → output TASK_COMPLETE immediately
   • Never add todos after execution has started
   • Todos are bookkeeping only — they never gate TASK_COMPLETE

5. SELF-CORRECT ONCE
   Error received → read it → adapt → one retry with different approach.
   Same error twice → BLOCKED.

6. NO REDUNDANT WORK
   Succeeded → don't repeat. Verified → don't re-verify. Done → TASK_COMPLETE.

════════════════════════════════════════════════════
WORKFLOW
════════════════════════════════════════════════════

  Phase 1 — Orient : glob/ls/read to understand scope (skip if task is obvious)
  Phase 2 — Execute: act and verify each step once
  Phase 3 — Confirm: deliverable exists and is correct (one check)
  Phase 4 — Done   : TASK_COMPLETE

  One file task? Skip Phase 1. Write it, read it back, TASK_COMPLETE.

════════════════════════════════════════════════════
SCHEDULED WAIT PROTOCOL
════════════════════════════════════════════════════

To pause until a future time:

  WAIT: <ISO_datetime>: <reason>
  Example: WAIT: 2026-03-01T09:00:00: Waiting for market open.

  • Must be a future datetime: YYYY-MM-DDTHH:MM:SS
  • Do NOT follow WAIT with TASK_COMPLETE

════════════════════════════════════════════════════
WHEN YOU'RE STUCK
════════════════════════════════════════════════════

Output exactly:

  BLOCKED:
  Reason: <one sentence>
  Failed operation: <tool name>
  Error received: <exact error>
  What I need: <specific requirement>

BLOCKED is a valid exit. Use it rather than looping forever.

════════════════════════════════════════════════════
TOOLS
════════════════════════════════════════════════════

  Files:      read, write, edit, glob, grep, ls, mkdir
  Shell:      shell (sandboxed — timeout 30 s, memory 512 MB by default)
  Git:        git_status, git_diff, git_add, git_commit, git_branch
  Todos:      todo_add, todo_complete, todo_update, todo_list
  Task State: task_state_update, task_state_get, task_reconcile
  Planning:   plan_complete_step
  Delegation: task (sub-agent for a single file — once per file maximum)

Shell sandbox backends (selected automatically):
  Windows + pywin32  → Job Object (process tree killed on handle close)
  Windows, no pywin32→ psutil tree-kill
  macOS / Linux      → process group + RLIMIT_AS + RLIMIT_CPU
"""

SUB_AGENT_SYSTEM_PROMPT = """You are a precise file-creation agent. One job: create the file exactly as specified.

YEAR: 2026. Modern standards. Don't mention the year.

Rules:
  • 2 sentences of reasoning max — then write immediately
  • write requires both "path" AND complete "content" — no placeholders
  • After writing: read it back once to verify
  • Verified correct → output TASK_COMPLETE and stop
  • Failed or wrong → output BLOCKED: <one line reason> and stop
  • Never loop. Never ask questions. Never plan.

TASK_COMPLETE means stop. Output it and nothing else follows.

Tools available: write, read, edit, ls, glob
"""

PLAN_MODE_PROMPT = """You are in PLAN MODE. Produce a concrete, actionable plan. Do not execute anything.

YEAR: 2026. Modern practices. Don't mention the year.

Think briefly (3 sentences max), then output the JSON immediately.

Consider before writing steps:
  • What is actually in scope?
  • What is the correct order and what has dependencies?
  • Where are the failure risks?
  • How is each step verified as done?

OUTPUT FORMAT — valid JSON only, no text before or after the JSON block:

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

After the JSON, output exactly this on its own line:
PLAN_APPROVED
"""


# =============================================================================
# SUB-AGENT EXECUTION  (shell intentionally excluded — sub-agents are file-only)
# =============================================================================

_SUB_AGENT_TOOL_NAMES = frozenset({"write", "read", "edit", "ls", "glob"})


def run_sub_agent(
    messages: List[Dict[str, Any]],
    workspace: Path,
    session_id: str,
    max_iterations: int,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    available_tools = [t for t in TOOL_SCHEMAS
                       if t["function"]["name"] in _SUB_AGENT_TOOL_NAMES]
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

_PLAN_TOOL_NAMES = frozenset({"ls", "read", "glob", "grep", "git_status"})


def run_plan_mode(task: str, workspace: Path) -> Optional[Dict[str, Any]]:
    Log.plan("Building execution plan…")
    messages = [
        {"role": "system", "content": PLAN_MODE_PROMPT},
        {"role": "user",   "content": f"Create plan for:\n\n{task}"},
    ]
    planning_tools = [t for t in TOOL_SCHEMAS
                      if t["function"]["name"] in _PLAN_TOOL_NAMES]

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
) -> Tuple[bool, PermissionMode]:
    """Execute all tool calls for one agent iteration.
    Returns (had_rename_op, updated_permission_mode).
    had_rename_op is always False — shell sandbox captures stdout rather than
    parsing it for rename operations; git tools handle their own rename logic.
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

    return False, current_permission_mode


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
