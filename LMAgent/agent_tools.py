#!/usr/bin/env python3
"""
agent_tools.py — Tool layer for LMAgent.

Contains: tool handlers, schemas, registry, vision support, git helpers.
LLM client, agent loops, and system prompts live in agent_llm.py.

Backward compatibility: everything that was previously importable from
agent_tools is still importable from agent_tools via the re-export block
at the bottom of this file.  No external files need to change.
"""

import base64
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
#
# Routes through run_sandboxed() so git executes inside the Docker container
# (or process-group fallback) rather than on the host shell.
# =============================================================================

def _git_safe(workspace: Path, cmd: str) -> Tuple[str, int]:
    ok, reason = Safety.validate_command(cmd, workspace)
    if not ok:
        raise ValueError(reason)
    return run_sandboxed(cmd=cmd, workspace=workspace)


# =============================================================================
# VISION SUPPORT
# =============================================================================

_vision_cache: Optional[bool] = None   # None = unchecked, True/False = result
_vision_lock = threading.Lock()

_VISION_MIME_MAP: Dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _detect_vision_support() -> bool:
    """
    Probe LM Studio's /api/v1/models endpoint to determine whether the currently
    loaded model has vision capability.  Result is cached for the process lifetime.

    Controlled by the VISION_ENABLED env var (read via Config):
      "true"  — always report vision available (skip the probe)
      "false" — always report no vision (skip the probe)
      "auto"  — probe LM Studio (default)

    Call vision_cache_invalidate() to force a fresh probe.
    """
    global _vision_cache
    with _vision_lock:
        if _vision_cache is not None:
            return _vision_cache

        mode = getattr(Config, "VISION_ENABLED", "auto").lower()
        if mode == "true":
            _vision_cache = True
            return True
        if mode == "false":
            _vision_cache = False
            return False

        # "auto" — probe LM Studio's native model list endpoint
        base = getattr(Config, "LLM_BASE_URL", "http://localhost:1234")
        try:
            resp = requests.get(
                f"{base}/api/v1/models",
                timeout=5,
                headers={"Authorization": f"Bearer {Config.LLM_API_KEY}"},
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            target = (Config.LLM_MODEL or "").lower()

            # First pass: try to match the configured model name exactly.
            for m in models:
                key = (m.get("key") or m.get("id") or "").lower()
                if target and target not in key:
                    continue
                if m.get("capabilities", {}).get("vision", False):
                    Log.info(f"Vision capability detected on configured model: {key}")
                    _vision_cache = True
                    return True

            # Second pass: accept any loaded vision-capable model as a fallback.
            for m in models:
                if m.get("capabilities", {}).get("vision", False):
                    key = m.get("key") or m.get("id") or "unknown"
                    Log.info(f"Vision fallback model found: {key}")
                    _vision_cache = True
                    return True

            Log.info("No vision-capable model found in LM Studio — vision tool disabled")
            _vision_cache = False

        except Exception as e:
            Log.warning(f"Vision probe failed ({e}) — disabling vision tool for this session")
            _vision_cache = False

        return _vision_cache


def vision_cache_invalidate() -> None:
    """Force a fresh vision-capability probe on the next tool assembly.

    Call this whenever the user loads a different model in LM Studio so the
    vision tool appears/disappears appropriately without restarting the server.
    """
    global _vision_cache
    with _vision_lock:
        _vision_cache = None
    Log.info("Vision cache invalidated — will re-probe on next tool call")


def tool_vision(
    workspace: Path,
    path: str,
    prompt: str = "Describe this image in detail.",
) -> Dict[str, Any]:
    """Send a workspace image to the loaded vision model and return its description.

    The tool validates the path, base64-encodes the image, and submits it to the
    OpenAI-compatible /v1/chat/completions endpoint using the image_url content
    type.  Only JPEG, PNG, GIF, and WebP files are accepted.

    Returns:
        {"success": True, "path": "...", "description": "<model output>"}
    or
        {"success": False, "error": "<reason>"}
    """
    if not _detect_vision_support():
        return {
            "success": False,
            "error": (
                "No vision-capable model is currently loaded in LM Studio. "
                "Load a VLM (e.g. LLaVA, Qwen-VL, Pixtral) and try again, "
                "or set VISION_ENABLED=true in .env to skip the check."
            ),
        }

    ok, err, fp = Safety.validate_path(workspace, path, must_exist=True)
    if not ok:
        return {"success": False, "error": err}
    if not fp.is_file():
        return {"success": False, "error": f"Not a file: {path}"}

    mime = _VISION_MIME_MAP.get(fp.suffix.lower())
    if not mime:
        supported = ", ".join(sorted(_VISION_MIME_MAP.keys()))
        return {
            "success": False,
            "error": (
                f"Unsupported image type '{fp.suffix}'. "
                f"Supported extensions: {supported}"
            ),
        }

    try:
        b64 = base64.b64encode(fp.read_bytes()).decode("utf-8")
    except Exception as e:
        return {"success": False, "error": f"Could not read image file: {e}"}

    payload: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": Config.TEMPERATURE,
    }
    if Config.LLM_MODEL:
        payload["model"] = Config.LLM_MODEL

    try:
        resp = requests.post(
            Config.LLM_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {Config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]
        return {
            "success":     True,
            "path":        str(fp.relative_to(workspace)).replace("\\", "/"),
            "description": description,
        }
    except Exception as e:
        return {"success": False, "error": f"Vision request failed: {e}"}


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
        tmp = fp.parent / (fp.name + ".tmp")
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
        tmp = fp.parent / (fp.name + ".tmp")
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
            search_files = [
                f for f in workspace.rglob("*")
                if f.is_file()
                and not any(p.startswith(".") for p in f.relative_to(workspace).parts)
            ]
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
      • Windows, no pywin32: psutil tree-kill on timeout.
      • macOS / Linux      : new process group (os.setsid) + RLIMIT_AS +
        RLIMIT_CPU — entire group killed on timeout or error.

    stdout and stderr are merged; output is capped at 32 KB.
    """
    ok, reason = Safety.validate_command(command, workspace)
    if not ok:
        return {"success": False, "error": f"Command blocked by safety check: {reason}"}

    timeout       = max(1, min(timeout, 120))
    max_memory_mb = max(64, min(max_memory_mb, 2048))

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
# TOOL HANDLERS — GIT  (sandboxed via _git_safe → run_sandboxed)
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
        resolved_path = ""
        if path:
            ok, err, resolved = Safety.validate_path(workspace, path)
            if not ok:
                return {"success": False, "error": f"Invalid diff path: {err}"}
            resolved_path = str(resolved.relative_to(workspace))
        cmd = "git diff"
        if staged:
            cmd += " --staged"
        if resolved_path:
            cmd += f" -- {_shell_quote(resolved_path)}"
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
# SUB-AGENT CONTEXT HELPER  (used by tool_task below)
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
    """Delegate file creation to an isolated sub-agent.

    Uses a lazy import of run_sub_agent / SUB_AGENT_SYSTEM_PROMPT from
    agent_llm to break the circular dependency:
        agent_llm imports agent_tools (schemas/handlers at module level)
        agent_tools lazy-imports agent_llm only when tool_task() is called
    """
    if not Config.ENABLE_SUB_AGENTS:
        return {"success": False, "error": "Sub-agents disabled"}

    # Lazy import — by call-time both modules are fully initialised.
    from agent_llm import run_sub_agent, SUB_AGENT_SYSTEM_PROMPT  # noqa: PLC0415

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

# TOOL_SCHEMAS is the full universe of possible tools.
# Never pass this directly to LLMClient.call() — use get_available_tools() instead,
# which filters out "vision" when no VLM is detected.

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

    # ── vision (conditionally included by get_available_tools) ───────────────
    {"type": "function", "function": {
        "name": "vision",
        "description": (
            "Read and analyse an image file from the workspace using the loaded "
            "vision model. Use this whenever the task involves an image, screenshot, "
            "diagram, chart, photo, or other visual file. Pass the workspace-relative "
            "path and an optional specific question about the image. "
            "Only available when a vision-capable model (VLM) is loaded in LM Studio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to the image file (jpg, png, gif, webp)",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "What to ask about the image. "
                        "Default: 'Describe this image in detail.'"
                    ),
                },
            },
            "required": ["path"],
        }}},
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
    # vision: conditionally available — registered so dispatch works
    "vision":             tool_vision,
}

# Tools that always require non-empty arguments — used by _parse_stream in agent_llm.
_REQUIRED_ARG_TOOLS = frozenset({
    "shell",
    "write", "read", "edit", "glob", "grep", "ls", "mkdir",
    "todo_add", "todo_complete", "todo_update", "plan_complete_step",
    "git_add", "git_commit", "git_branch", "git_diff",
    "task",
    "vision",
})


# =============================================================================
# DYNAMIC TOOL LIST  ← use this everywhere instead of TOOL_SCHEMAS directly
# =============================================================================

def get_available_tools(
    extra_schemas: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return the tool list to pass to LLMClient.call() for the current session.

    The "vision" tool is included only when _detect_vision_support() returns True.

    Args:
        extra_schemas: Additional tool schemas to merge in (e.g. from MCP servers).
                       Extra schemas are deduplicated by name; extras win on collision.

    Usage:
        available_tools = get_available_tools(extra_schemas=mcp_manager.get_all_tools())

        # Force a fresh vision probe after user switches models:
        vision_cache_invalidate()
        available_tools = get_available_tools(...)
    """
    has_vision = _detect_vision_support()

    tools = [
        t for t in TOOL_SCHEMAS
        if t["function"]["name"] != "vision" or has_vision
    ]

    if extra_schemas:
        existing_names = {t["function"]["name"] for t in tools}
        for schema in extra_schemas:
            name = (schema.get("function") or {}).get("name", "")
            if name and name not in existing_names:
                tools.append(schema)
                existing_names.add(name)

    return tools


# =============================================================================
# BACKWARD-COMPAT RE-EXPORTS
#
# Everything that lived in agent_tools.py before the split is still importable
# from agent_tools.  External files need zero changes.
#
# HOW THE CIRCULAR IMPORT IS AVOIDED:
#   1. Python starts loading agent_tools (this file).
#   2. All tool handlers, TOOL_SCHEMAS, TOOL_HANDLERS, get_available_tools
#      are fully defined above this point.
#   3. This block triggers agent_llm to load.
#   4. agent_llm does `from agent_tools import ...` — Python finds agent_tools
#      already in sys.modules (partially loaded) but with everything it needs
#      already defined (steps 1-2 above).  Import succeeds.
#   5. agent_llm finishes loading.
#   6. The names below are bound in this module's namespace.
#
# tool_task uses a *call-time* lazy import of run_sub_agent/SUB_AGENT_SYSTEM_PROMPT
# so there is no module-level cycle from that direction either.
# =============================================================================

from agent_llm import (  # noqa: E402, F401
    LLMClient,
    detect_completion,
    should_ask_permission,
    ask_permission,
    SYSTEM_PROMPT,
    SUB_AGENT_SYSTEM_PROMPT,
    PLAN_MODE_PROMPT,
    run_sub_agent,
    run_plan_mode,
    _execute_tool,
    _process_tool_calls,
    _HeaderStreamCb,
)
