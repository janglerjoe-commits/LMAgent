#!/usr/bin/env python3
"""
agent_main.py — Main entrypoint for LMAgent.
FIXES: [1] truncation key mismatch, [2/3] double retry loop, [4] .env injection, [5] dead _hold_lock param, [6] module attr patching, [7] unreachable None guard, [8] _running_lock scope, [9] soul loaded twice, [10] scheduler stdout bleed, [11] scheduler UX headers, [12] scheduler output mode, [13] scheduler worker never updated current_session_id so follow-up messages always started a fresh session instead of continuing.
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent_core import (
    VERSION, Config, PermissionMode, Colors, Log,
    colored, rainbow_text,
    InstanceLock, cleanup_resources,
    AgentState, AgentEvent, AgentResult,
    StateManager, SessionManager,
    TodoManager, PlanManager, TaskStateManager,
    WaitState, detect_wait,
    LoopDetector, MCPManager,
    ProjectConfig, SoulConfig,
    compact_messages,
    close_shell_session,
    _instance_lock as _core_instance_lock,
)
from agent_tools import (
    TOOL_SCHEMAS, TOOL_HANDLERS,
    LLMClient, detect_completion,
    should_ask_permission, ask_permission,
    set_tool_context, get_current_context,
    run_plan_mode, run_sub_agent,
    _process_tool_calls, _unpack_tc, _parse_tool_args,
    _HeaderStreamCb,
    SYSTEM_PROMPT, SUB_AGENT_SYSTEM_PROMPT,
    get_available_tools,
    vision_cache_invalidate,
)

import agent_core as _core_mod


_repl_prompt_fn: Optional[Callable[[], None]] = None
_worker_stdout_lock: threading.Lock = threading.Lock()


# =============================================================================
# WORKSPACE SECURITY
# =============================================================================

def _lock_workspace(raw_path: str) -> Path:
    try:
        resolved = Path(raw_path).expanduser().resolve()
    except Exception as e:
        print(colored(f"\n[!] Could not resolve workspace path {raw_path!r}: {e}", Colors.RED))
        sys.exit(1)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(colored(f"\n[!] Cannot create workspace {resolved}: {e}", Colors.RED))
        sys.exit(1)
    probe = resolved / ".lmagent" / ".write_probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
    except Exception as e:
        print(colored(f"\n[!] Workspace {resolved} is not writable: {e}", Colors.RED))
        sys.exit(1)
    Config.WORKSPACE = str(resolved)
    return resolved


def _sanitize_env_value(raw: str) -> str:
    cleaned = "".join(ch for ch in raw if ch >= " " and ch != "\x7f")
    cleaned = cleaned.replace('"', "")
    return cleaned


# =============================================================================
# WORKSPACE PICKER
# =============================================================================

def _pick_workspace(args_workspace: Optional[str]) -> str:
    if args_workspace:
        return args_workspace
    raw = os.environ.get("WORKSPACE", "").strip()
    if raw:
        return raw
    default = Config.WORKSPACE
    print(colored("\n┌─ Workspace Setup ──────────────────────────────────────────────┐", Colors.CYAN))
    print(colored("│  No workspace configured.                                      │", Colors.CYAN))
    print(colored(f"│  Default: {default:<54}│", Colors.CYAN))
    print(colored("│                                                                │", Colors.CYAN))
    print(colored("│  ⚠  The agent will ONLY be able to read and write files        │", Colors.YELLOW))
    print(colored("│     inside this directory.  Choose carefully.                  │", Colors.YELLOW))
    print(colored("└────────────────────────────────────────────────────────────────┘", Colors.CYAN))
    try:
        choice = input(colored("  Press Enter to use the default, or type a path: ", Colors.YELLOW)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        choice = ""
    chosen = choice if choice else default
    if chosen != default:
        script_dir = Path(__file__).parent
        env_file   = script_dir / ".env"
        try:
            save = input(colored(f"  Save this path to {env_file.name} for future runs? [y/N]: ", Colors.YELLOW)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            save = ""
        if save in ("y", "yes"):
            try:
                lines: List[str] = []
                if env_file.exists():
                    lines = env_file.read_text(encoding="utf-8").splitlines()
                safe_chosen = _sanitize_env_value(chosen)
                found = False
                for i, line in enumerate(lines):
                    if line.strip().startswith("WORKSPACE="):
                        lines[i] = f'WORKSPACE="{safe_chosen}"'
                        found = True
                        break
                if not found:
                    lines.append(f'WORKSPACE="{safe_chosen}"')
                env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(colored(f"  ✓ Saved to {env_file}", Colors.GREEN))
            except Exception as e:
                print(colored(f"  Could not write {env_file}: {e}", Colors.YELLOW))
    return chosen


# =============================================================================
# SLASH COMMANDS
# =============================================================================

@dataclass
class SlashCommand:
    name:        str
    description: str
    handler:     Callable


def cmd_help(workspace: Path, **_) -> str:
    return colored("""
╭──────────────────────────────────────────────────────╮
│                   Available Commands                 │
├──────────────────────────────────────────────────────┤
│  /help        This help screen                       │
│  /sessions    List recent sessions                   │
│  /mode <lvl>  Set permission: manual / normal / auto │
│  /plan        Show the current plan                  │
│  /todo        Show todo list                         │
│  /status      Show agent + MCP status                │
│  /soul        Show loaded soul / personality         │
│  /new         Start a completely fresh session       │
│  /session     Show current session ID                │
│  quit         Exit                                   │
╰──────────────────────────────────────────────────────╯
""", Colors.CYAN)


def cmd_sessions(workspace: Path, **_) -> str:
    sessions = SessionManager(workspace).list_recent(10)
    if not sessions:
        return colored("  No sessions found.", Colors.GRAY)
    status_colors = {
        "completed": Colors.GREEN, "waiting": Colors.MAGENTA,
        "active":    Colors.CYAN,  "error":   Colors.RED,
    }
    lines = [colored("\nRecent Sessions:", Colors.CYAN, bold=True)]
    for s in sessions:
        sc     = status_colors.get(s["status"], Colors.YELLOW)
        parent = f"  ← {s['parent'][:8]}" if s.get("parent") else ""
        lines.append(
            f"  {colored(s['id'], Colors.CYAN)}  "
            f"{colored(s['status'], sc):<12}  "
            f"{s['task'][:40]}{parent}"
        )
    return "\n".join(lines)


def cmd_mode(workspace: Path, mode: str = "", **_) -> str:
    if not mode:
        return "Usage: /mode <manual|normal|auto>"
    try:
        PermissionMode(mode)
        return colored(f"  ✓ Permission mode → {mode}", Colors.GREEN)
    except ValueError:
        return colored(f"  Unknown mode '{mode}'. Choose: manual, normal, auto", Colors.RED)


def cmd_plan(workspace: Path, session_id: str = "", **_) -> str:
    if not session_id:
        return colored("  No active session.", Colors.GRAY)
    PlanManager(workspace, session_id).display()
    return ""


def cmd_todo(workspace: Path, session_id: str = "", **_) -> str:
    if not session_id:
        return colored("  No active session.", Colors.GRAY)
    TodoManager(workspace, session_id).display()
    return ""


def cmd_status(workspace: Path, mcp_manager=None, session_id: str = "", **_) -> str:
    lines = [
        colored("\n  ● Agent Status", Colors.CYAN, bold=True),
        colored("  " + "─" * 48, Colors.CYAN),
        f"  Workspace  : {workspace}",
        f"  LLM        : {Config.LLM_URL}",
        f"  Session    : {session_id[-8:] if session_id else 'none'}",
        f"  Permissions: {Config.PERMISSION_MODE}",
        f"  Shell lock : {'workspace-only' if Config.SHELL_WORKSPACE_ONLY else 'UNLOCKED (destructive still blocked)'}",
    ]
    if mcp_manager:
        status = mcp_manager.get_status()
        if status:
            lines.append("\n  MCP Servers:")
            for name, ok in status.items():
                mark = colored("✓", Colors.GREEN) if ok else colored("✗", Colors.RED)
                lines.append(f"    {mark} {name}")
    lines += [
        "\n  Features:",
        f"    Sub-agents     : {'on' if Config.ENABLE_SUB_AGENTS     else 'off'}",
        f"    Todos          : {'on' if Config.ENABLE_TODO_TRACKING   else 'off'}",
        f"    Plans          : {'on' if Config.ENABLE_PLAN_ENFORCEMENT else 'off'}",
        f"    Thinking model : {'on' if Config.THINKING_MODEL         else 'off'}",
        f"    Context memory : on  (use /new to reset)",
    ]
    return "\n".join(lines)


def cmd_soul(workspace: Path, **_) -> str:
    soul  = SoulConfig.load(workspace)
    lines = [
        colored("\n  ● Soul Config  (.soul.md)", Colors.MAGENTA, bold=True),
        colored("  " + "─" * 48, Colors.MAGENTA),
    ] + [f"  {line}" for line in soul.splitlines()] + [
        colored("\n  Edit .soul.md in your workspace to customise personality.", Colors.GRAY)
    ]
    return "\n".join(lines)


def cmd_new(workspace: Path, **_) -> str:
    return "__RESET__"


def cmd_session(workspace: Path, session_id: str = "", **_) -> str:
    if not session_id:
        return colored("  No active session — send your first task to create one.", Colors.GRAY)
    return colored(f"  Current session: {session_id}", Colors.CYAN)


SLASH_COMMANDS: Dict[str, SlashCommand] = {
    "/help":     SlashCommand("/help",     "Help",            cmd_help),
    "/sessions": SlashCommand("/sessions", "List sessions",   cmd_sessions),
    "/mode":     SlashCommand("/mode",     "Set permission",  cmd_mode),
    "/plan":     SlashCommand("/plan",     "Show plan",       cmd_plan),
    "/todo":     SlashCommand("/todo",     "Show todos",      cmd_todo),
    "/status":   SlashCommand("/status",   "Agent status",    cmd_status),
    "/soul":     SlashCommand("/soul",     "Show soul",       cmd_soul),
    "/new":      SlashCommand("/new",      "Fresh session",   cmd_new),
    "/session":  SlashCommand("/session",  "Current session", cmd_session),
}


# =============================================================================
# run_scheduler()
# =============================================================================

def run_scheduler(
    workspace: Path,
    poll_interval: Optional[int] = None,
    soul: str = "",
    # FIX [13]: called with the completed session_id after each worker finishes
    # so the REPL's current_session_id stays in sync and follow-up messages
    # resume the same session instead of starting a fresh one.
    session_callback: Optional[Callable[[str], None]] = None,
) -> None:
    _running_lock: threading.Lock = threading.Lock()
    _running:      set            = set()
    _wake_event:   threading.Event = threading.Event()

    interval = poll_interval if poll_interval is not None else Config.SCHEDULER_POLL_INTERVAL
    inbox    = workspace / ".lmagent" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    session_mgr = SessionManager(workspace)
    state_mgr   = StateManager(workspace)

    Log.info(f"Scheduler started — polling every {interval}s")
    Log.info(f"Workspace : {workspace}")
    Log.info(f"Inbox     : {inbox}  (drop a .task file to submit work)")

    def _is_running(key: str) -> bool:
        with _running_lock:
            return key in _running

    def _mark_done(key: str) -> None:
        with _running_lock:
            _running.discard(key)

    def _worker(task: str, resume_sid: Optional[str], run_key: str) -> None:
        sid_label = (resume_sid or run_key)[:8]

        with _worker_stdout_lock:
            sys.stdout.write(
                "\n" + colored(
                    f"╔══ [scheduler] ▶  session {sid_label} waking"
                    f" ════════════════════════════════╗",
                    Colors.MAGENTA,
                ) + "\n\n"
            )
            sys.stdout.flush()

        try:
            result = run_agent(
                task,
                workspace,
                permission_mode=PermissionMode.AUTO,
                resume_session=resume_sid,
                plan_first=False,
                mode="interactive",
                soul=soul,
            )

            with _worker_stdout_lock:
                status_color = (
                    Colors.GREEN  if result.status == "completed" else
                    Colors.YELLOW if result.status == "waiting"   else
                    Colors.RED
                )
                sys.stdout.write(
                    "\n" + colored(
                        f"╚══ [scheduler] ✓  session {result.session_id[:8]}"
                        f" → {result.status}"
                        f" ══════════════════════════════════╝",
                        status_color,
                    ) + "\n"
                )
                if result.final_answer and result.status != "completed":
                    preview = result.final_answer[:300].replace("\n", " ")
                    sys.stdout.write(colored(f"  {preview}", Colors.CYAN) + "\n")
                sys.stdout.flush()

            # FIX [13]: tell the REPL which session just finished so the next
            # user message continues it rather than starting from scratch.
            if session_callback and result.status == "completed":
                session_callback(result.session_id)

            if _repl_prompt_fn is not None:
                _repl_prompt_fn()

            if result.status == "waiting":
                Log.wait(
                    f"[scheduler] session {result.session_id} sleeping "
                    f"until {result.wait_until}"
                )
                _wake_event.set()

        except Exception as e:
            with _worker_stdout_lock:
                Log.error(f"[scheduler] worker error (session={sid_label}): {e}")
            if _repl_prompt_fn is not None:
                _repl_prompt_fn()
        finally:
            _mark_done(run_key)

    def _spawn(task: str, resume_sid: Optional[str], run_key: str) -> None:
        with _running_lock:
            if run_key in _running:
                return
            _running.add(run_key)
        threading.Thread(
            target=_worker,
            args=(task, resume_sid, run_key),
            daemon=True,
            name=f"sched-{run_key[:12]}",
        ).start()

    try:
        while True:
            try:
                next_wake_secs: Optional[float] = None

                for task_file in sorted(inbox.glob("*.task")):
                    try:
                        task_text = task_file.read_text(encoding="utf-8").strip()
                        if not task_text:
                            task_file.unlink(missing_ok=True)
                            continue
                        run_key = f"inbox-{task_file.stem}"
                        Log.info(f"[scheduler] inbox: {task_text[:80]}")
                        task_file.unlink(missing_ok=True)
                        _spawn(task_text, resume_sid=None, run_key=run_key)
                    except Exception as e:
                        Log.error(f"[scheduler] failed to read inbox task {task_file.name}: {e}")
                        try:
                            task_file.rename(task_file.with_suffix(".failed"))
                        except Exception:
                            pass

                for session in session_mgr.list_recent(200):
                    if session.get("status") != "waiting":
                        continue
                    sid = session["id"]
                    if _is_running(sid):
                        continue

                    saved_state = state_mgr.load(sid)

                    if not saved_state or not saved_state.wait_state:
                        Log.warning(
                            f"[scheduler] [{sid[:8]}] stuck in 'waiting' with "
                            f"no wait_state — rescuing back to 'idle'"
                        )
                        try:
                            data = session_mgr.load(sid)
                            if data:
                                msgs, meta = data
                                meta["status"] = "idle"
                                session_mgr.save(sid, msgs, meta)
                        except Exception as rescue_err:
                            Log.error(f"[scheduler] [{sid[:8]}] rescue failed: {rescue_err}")
                        continue

                    ws = WaitState.from_dict(saved_state.wait_state)

                    if not ws.is_ready():
                        try:
                            remaining = (
                                datetime.fromisoformat(ws.resume_after) - datetime.now()
                            ).total_seconds()
                            if remaining > 0:
                                next_wake_secs = (
                                    remaining if next_wake_secs is None
                                    else min(next_wake_secs, remaining)
                                )
                            Log.info(
                                f"[scheduler] [{sid[:8]}] sleeping — "
                                f"{max(0, remaining):.0f}s left"
                                f"  ({ws.reason[:50]})"
                            )
                        except Exception:
                            pass
                        continue

                    Log.info(f"[scheduler] [{sid[:8]}] waking — {ws.reason}")
                    _spawn("Continue from scheduled wake-up", resume_sid=sid, run_key=sid)

            except Exception as poll_err:
                Log.error(f"[scheduler] poll error: {poll_err}")

            sleep_for = (
                max(0.5, next_wake_secs + 0.25)
                if next_wake_secs is not None and next_wake_secs < interval
                else interval
            )
            if sleep_for < interval:
                Log.info(
                    f"[scheduler] sleeping {sleep_for:.1f}s "
                    f"(next wake-up in ~{next_wake_secs:.0f}s)"
                )

            _wake_event.clear()
            _wake_event.wait(timeout=sleep_for)

    except KeyboardInterrupt:
        Log.info("[scheduler] shutting down — waiting for active agents…")
        deadline = time.time() + 10
        while True:
            with _running_lock:
                if not _running:
                    break
            if time.time() > deadline:
                Log.warning("[scheduler] timeout — some workers may still be running")
                break
            time.sleep(0.5)
        Log.info("[scheduler] exited.")


# =============================================================================
# MAIN AGENT
# =============================================================================

def run_agent(
    task: str,
    workspace: Path,
    permission_mode: PermissionMode = PermissionMode.NORMAL,
    resume_session: Optional[str] = None,
    plan_first: bool = False,
    mode: str = "interactive",
    event_callback: Optional[Callable[[AgentEvent], None]] = None,
    soul: str = "",
    whisper_fn: Optional[Callable[[], Optional[str]]] = None,
) -> AgentResult:
    expected_ws = Path(Config.WORKSPACE).resolve()
    if workspace.resolve() != expected_ws:
        raise ValueError(
            f"run_agent() called with workspace={workspace!r} but "
            f"Config.WORKSPACE is locked to {expected_ws!r}. "
            "Always use the Path returned by _lock_workspace()."
        )

    if mode == "output":
        Log.set_silent(True)
        permission_mode = PermissionMode.AUTO

    events:            List[AgentEvent] = []
    final_answer       = ""
    iteration          = 0
    caller_forced_auto = (permission_mode == PermissionMode.AUTO)

    _base_stream_cb: Optional[Callable[[str], None]] = None
    if mode == "interactive":
        def _base_stream_cb(token: str):
            sys.stdout.write(token)
            sys.stdout.flush()

    _stream_cb = _HeaderStreamCb(_base_stream_cb, mode)

    def emit(event_type: str, data: Dict[str, Any]) -> None:
        ev = AgentEvent(type=event_type, data=data)
        events.append(ev)
        if event_callback:
            event_callback(ev)
        if mode != "interactive":
            return
        msg = data.get("message", data.get("error", ""))
        if   event_type == "log":         Log.info(msg)
        elif event_type == "warning":     Log.warning(msg)
        elif event_type == "error":       Log.error(msg)
        elif event_type == "tool_call":
            Log.tool(data.get("name", ""), data.get("args_preview", ""))
        elif event_type == "tool_result":
            (Log.success if data.get("success") else Log.error)(
                f"{'✓' if data.get('success') else '✗'} {data.get('name', '')}"
            )
        elif event_type == "complete":    Log.success(f"Done — {data.get('reason', '')}")
        elif event_type == "iteration":   Log.info(f"\nIteration {data.get('n')}/{Config.MAX_ITERATIONS}")
        elif event_type == "waiting":     Log.wait(f"Sleeping until {data.get('resume_after')}: {data.get('reason')}")

    session_mgr = SessionManager(workspace)
    state_mgr   = StateManager(workspace)

    if resume_session:
        emit("log", {"message": f"Resuming session: {resume_session}"})
        session_data = session_mgr.load(resume_session)
        if not session_data:
            emit("error", {"message": f"Session not found: {resume_session}"})
            return AgentResult(status="error", final_answer="Session not found.",
                               events=events, session_id=resume_session, iterations=0)

        messages, metadata = session_data
        iteration          = metadata.get("iterations", 0)
        session_id         = resume_session

        saved_state = state_mgr.load(session_id)
        if saved_state:
            detector = LoopDetector.from_dict(saved_state.loop_detector_state)
            if not caller_forced_auto:
                permission_mode = PermissionMode(saved_state.permission_mode)
            emit("log", {"message": f"State restored from iteration {saved_state.iteration}"})
        else:
            detector = LoopDetector()

        if saved_state and saved_state.wait_state:
            ws         = WaitState.from_dict(saved_state.wait_state)
            resume_msg = ws.context_on_resume.replace("{current_time}", datetime.now().isoformat())
            messages.append({"role": "user", "content": resume_msg})
            emit("log", {"message": f"Waking from sleep. Reason: {ws.reason}"})
            saved_state.wait_state = None
            state_mgr.save(session_id, saved_state)
            session_mgr.save(session_id, messages, {
                **metadata, "status": "active", "iterations": iteration,
            })
        elif (task and task.strip()
              and task not in ("Continue from previous session",
                               "Continue from scheduled wake-up")):
            messages.append({"role": "user", "content": task})
            emit("log", {"message": f"User follow-up: {task[:80]}"})

        emit("log", {"message": f"Resumed at iteration {iteration}"})

    else:
        emit("log", {"message": f"Task: {task[:100]}"})
        plan_data = None

        if plan_first:
            plan_data = run_plan_mode(task, workspace)
            if plan_data:
                if mode == "interactive":
                    print(colored("\n📋 Proposed Plan:", Colors.BLUE, bold=True))
                    print(colored("─" * 60, Colors.BLUE))
                    print(json.dumps(plan_data, indent=2))
                    print(colored("─" * 60, Colors.BLUE))
                    if input(colored("\nProceed? [y/n]: ", Colors.YELLOW)).lower().strip() \
                            not in ("y", "yes"):
                        return AgentResult(status="cancelled", final_answer="Plan not approved.",
                                           events=events, session_id="", iterations=0)
                    Log.plan("Plan approved — starting execution")
                emit("log", {"message": "Plan approved"})

        session_id = session_mgr.create(task)

        soul_section = (
            f"════════════════════════════════════════════════════\n"
            f"YOUR PERSONALITY\n"
            f"════════════════════════════════════════════════════\n\n"
            f"{soul}\n"
        ) if soul else ""

        now = datetime.now()
        final_system_prompt = (
            SYSTEM_PROMPT.replace("{soul_section}", soul_section)
            + f"\n\nCURRENT DATE/TIME: {now.strftime('%A, %Y-%m-%d %H:%M:%S')}"
        )

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user",   "content": task},
        ]

        project_cfg = ProjectConfig.load(workspace)
        if project_cfg:
            messages.insert(1, {"role": "system", "content": f"PROJECT CONFIG:\n\n{project_cfg}"})

        detector  = LoopDetector()
        iteration = 0

    todo_mgr       = TodoManager(workspace, session_id)
    plan_mgr       = PlanManager(workspace, session_id)
    task_state_mgr = TaskStateManager(workspace, session_id)
    set_tool_context(workspace, session_id, todo_mgr, plan_mgr, task_state_mgr,
                     messages, mode=mode, stream_callback=_base_stream_cb)

    if resume_session:
        loaded = task_state_mgr.load()
        if loaded:
            emit("log", {"message": (f"Task state loaded: "
                                     f"{loaded.processed_count}/{loaded.total_count} processed")})

    if not resume_session and plan_first and plan_data:
        plan_mgr.create(plan_data)

    mcp_manager     = MCPManager(workspace)
    mcp_manager.load_servers()
    mcp_tools       = mcp_manager.get_all_tools()
    available_tools = get_available_tools(extra_schemas=mcp_tools)

    current_permission_mode = permission_mode

    def _save_state(wait_state: Optional[WaitState] = None) -> None:
        state_mgr.save(session_id, AgentState(
            session_id=session_id,
            iteration=iteration,
            loop_detector_state=detector.to_dict(),
            permission_mode=current_permission_mode.value,
            current_plan_step=plan_mgr.current_step_id,
            last_checkpoint=datetime.now().isoformat(),
            wait_state=wait_state.to_dict() if wait_state else None,
        ))

    def _save_session(status: str) -> None:
        session_mgr.save(session_id, messages, {
            "id":              session_id,
            "task":            task[:200],
            "iterations":      iteration,
            "status":          status,
            "permission_mode": current_permission_mode.value,
        })

    def _inject_context() -> None:
        nonlocal messages
        if Config.ENABLE_PLAN_ENFORCEMENT and plan_mgr.plan:
            ctx = plan_mgr.get_context()
            if ctx:
                messages = [m for m in messages
                            if "ACTIVE PLAN:" not in str(m.get("content", ""))]
                messages.insert(1, {"role": "system", "content": ctx})
                nxt = plan_mgr.get_next_step()
                if nxt and plan_mgr.current_step_id != nxt["id"]:
                    plan_mgr.start_step(nxt["id"])

        if Config.ENABLE_TODO_TRACKING:
            ctx = todo_mgr.get_context()
            if ctx:
                messages = [m for m in messages
                            if "YOUR TODO LIST:" not in str(m.get("content", ""))]
                insert_pos = 2 if plan_mgr.plan else 1
                messages.insert(insert_pos, {"role": "system", "content": ctx})

        if task_state_mgr.current_state:
            messages = [m for m in messages
                        if "[TASK STATE - DO NOT SUMMARIZE]"
                        not in str(m.get("content", ""))]
            insert_pos = 1 + bool(plan_mgr.plan) + bool(todo_mgr.get_context())
            messages.insert(insert_pos, task_state_mgr.current_state.to_message())

        get_current_context()["messages"] = messages

    try:
        while iteration < Config.MAX_ITERATIONS:
            iteration += 1
            emit("iteration", {"n": iteration, "max": Config.MAX_ITERATIONS})

            loop_msg = detector.check(iteration)
            if loop_msg:
                emit("warning", {"message": loop_msg})
                if "stopped responding" in loop_msg:
                    emit("error", {"message": "Agent unresponsive — aborting"})
                    return AgentResult(status="error",
                                       final_answer=f"Agent stuck: {loop_msg}",
                                       events=events, session_id=session_id,
                                       iterations=iteration)
                messages.append({
                    "role": "user",
                    "content": f"⚠️ {loop_msg}\nIf the task is done, say TASK_COMPLETE.",
                })

            if iteration % 5 == 0:
                _save_state()

            _inject_context()
            messages = compact_messages(messages)
            _stream_cb.reset()

            _effective_stream_cb = _stream_cb if mode == "interactive" else None

            response = LLMClient.call(messages, available_tools,
                                      stream_callback=_effective_stream_cb)

            if "error" in response:
                emit("error", {"message": f"LLM error: {response['error']}"})
                return AgentResult(
                    status="error",
                    final_answer=f"LLM error: {response['error']}",
                    events=events, session_id=session_id, iterations=iteration,
                )

            content    = response.get("content", "")
            tool_calls = response.get("tool_calls") or []

            if not content and not tool_calls:
                detector.track_empty()
                continue

            if content:
                final_answer = content

            wait_state = detect_wait(content) if content else None
            if wait_state:
                emit("waiting", {"reason": wait_state.reason, "resume_after": wait_state.resume_after})
                _save_state(wait_state=wait_state)
                _save_session("waiting")
                if mode == "interactive":
                    print(colored(
                        f"\n⏸  Going to sleep until {wait_state.resume_after}\n"
                        f"   Reason     : {wait_state.reason}\n"
                        f"   Session ID : {session_id}\n"
                        f"   The scheduler will wake this automatically.\n",
                        Colors.MAGENTA,
                    ))
                return AgentResult(
                    status="waiting",
                    final_answer=(f"Waiting until {wait_state.resume_after}. "
                                  f"Reason: {wait_state.reason}"),
                    events=events, session_id=session_id, iterations=iteration,
                    wait_until=wait_state.resume_after,
                )

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content:    assistant_msg["content"]    = content
            if tool_calls: assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if tool_calls:
                current_permission_mode = _process_tool_calls(
                    tool_calls, workspace, available_tools, detector,
                    iteration, mcp_manager, messages, current_permission_mode,
                    emit=emit,
                )

            whisper_injected = False
            if whisper_fn:
                nudge = whisper_fn()
                if nudge:
                    emit("log", {"message": f"whisper injected: {nudge[:60]}"})
                    messages.append({
                        "role":    "system",
                        "content": f"[User nudge - mid-run instruction]: {nudge}",
                    })
                    whisper_injected = True

            had_truncation = any(tc.get("_truncated") for tc in tool_calls)
            if had_truncation:
                emit("warning", {"message": "Truncated tool call — skipping completion check, forcing retry"})
                continue

            is_complete, reason = detect_completion(content, bool(tool_calls))

            if is_complete and whisper_injected:
                emit("log", {"message": "whisper pending — deferring completion for one iteration"})
                is_complete = False

            if not is_complete and reason == "Asking for input" and not tool_calls:
                emit("log", {"message": "Conversational reply detected — closing cleanly"})
                is_complete = True
                reason      = "Conversational reply (awaiting next user message)"

            if is_complete and task_state_mgr.current_state:
                s = task_state_mgr.current_state
                if (s.completion_gate == "processed == total"
                        and s.processed_count != s.total_count):
                    emit("warning", {"message": f"Completion gate not met: {s.processed_count}/{s.total_count}"})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"⚠️ Cannot complete: {s.processed_count}/{s.total_count} "
                            "items processed. Finish remaining items first."
                        ),
                    })
                    is_complete = False

            if Config.ENABLE_PLAN_ENFORCEMENT and plan_mgr.plan and plan_mgr.is_complete():
                is_complete, reason = True, "Plan completed"

            if is_complete:
                emit("complete", {"reason": reason, "answer": final_answer})
                _save_session("idle")
                task_state_mgr.clear()
                if mode == "interactive":
                    print(colored(
                        "\n✅ Done. Ask a follow-up or give a new task.\n"
                        "   /new to start a fresh session.\n",
                        Colors.GREEN,
                    ))
                return AgentResult(
                    status="completed", final_answer=final_answer,
                    events=events, session_id=session_id, iterations=iteration,
                )

            if Config.AUTO_SAVE_SESSION and iteration % 5 == 0:
                _save_session("active")

        emit("warning", {"message": f"Reached max iterations ({Config.MAX_ITERATIONS})"})
        return AgentResult(
            status="max_iterations",
            final_answer=final_answer or "Max iterations reached without completing.",
            events=events, session_id=session_id, iterations=iteration,
        )

    except KeyboardInterrupt:
        emit("log", {"message": "Interrupted"})
        _save_state()
        if Config.AUTO_SAVE_SESSION:
            _save_session("interrupted")
        return AgentResult(status="interrupted",
                           final_answer=final_answer or "Interrupted.",
                           events=events, session_id=session_id, iterations=iteration)

    except Exception as e:
        emit("error", {"message": f"Fatal error: {e}"})
        traceback.print_exc()
        return AgentResult(status="error", final_answer=f"Fatal error: {e}",
                           events=events, session_id=session_id, iterations=iteration)

    finally:
        mcp_manager.close_all()
        close_shell_session()
        if mode == "output":
            Log.set_silent(False)


# =============================================================================
# BANNER
# =============================================================================

def print_banner() -> None:
    banner = r"""
      ___           ___           ___           ___           ___       ___     
     /\  \         /\  \         /\  \         /\  \         /\__\     /\__\    
    /::\  \       /::\  \       /::\  \       /::\  \       /:/  /    /::|  |   
   /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/  /    /:|:|  |   
  /:/  \:\  \   /:/  \:\  \   /::\~\:\  \   /::\~\:\  \   /:/  /    /:/|:|__|__ 
 /:/__/ \:\__\ /:/__/ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/__/    /:/ |::::\__\
 \:\  \  \/__/ \:\  \ /:/  / \/_|::\/:/  / \:\~\:\ \/__/ \:\  \    \/__/~~/:/  /
  \:\  \        \:\  /:/  /     |:|::/  /   \:\ \:\__\    \:\  \         /:/  / 
   \:\  \        \:\/:/  /      |:|\/__/     \:\ \/__/     \:\  \       /:/  /  
    \:\__\        \::/  /       |:|  |        \:\__\        \:\__\     /:/  /   
     \/__/         \/__/         \|__|         \/__/         \/__/     \/__/    
    """
    print(rainbow_text(banner, bold=True))


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

_EXIT_CODES: Dict[str, int] = {
    "completed":      0,
    "cancelled":      0,
    "waiting":        0,
    "max_iterations": 2,
    "interrupted":    130,
    "error":          1,
}


def main() -> int:
    global _repl_prompt_fn

    parser = argparse.ArgumentParser(
        description=f"LMAgent v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "SCHEDULER MODE (24/7 service):\n"
            "  python agent_main.py --scheduler\n\n"
            "SUBMIT TASK TO RUNNING SCHEDULER:\n"
            "  python agent_main.py --submit \"your task\"\n\n"
            "SOUL / PERSONALITY:\n"
            "  Create <workspace>/.soul.md to define the agent's personality.\n\n"
            "WORKSPACE:\n"
            "  Set via --workspace, the WORKSPACE env var, or a .env file\n"
            "  next to this script containing:  WORKSPACE=\"/your/path\"\n\n"
            "SECURITY:\n"
            "  The agent is sandboxed to the workspace directory.\n"
            "  Shell commands that reference paths outside the workspace are\n"
            "  blocked by default (SHELL_WORKSPACE_ONLY=true).\n"
            "  Destructive operations outside the workspace are ALWAYS blocked.\n"
        ),
    )
    parser.add_argument("task",                 nargs="?",  help="Task to execute")
    parser.add_argument("--workspace",                      help="Workspace directory")
    parser.add_argument("--resume",             metavar="SESSION_ID", help="Resume a previous session")
    parser.add_argument("--plan",               action="store_true",  help="Create an execution plan before running")
    parser.add_argument("--list-sessions",      action="store_true",  help="List recent sessions")
    parser.add_argument("--permission",         choices=["manual", "normal", "auto"], help="Permission mode")
    parser.add_argument("--no-mcp",             action="store_true",  help="Disable MCP servers")
    parser.add_argument("--no-sub-agents",      action="store_true",  help="Disable sub-agents")
    parser.add_argument("--scheduler",          action="store_true",  help="Run as a background scheduler daemon")
    parser.add_argument("--scheduler-interval", type=int, default=None, metavar="SECONDS",
                        help=f"Scheduler poll interval (default {Config.SCHEDULER_POLL_INTERVAL}s)")
    parser.add_argument("--submit",             metavar="TASK", help="Submit a task to a running scheduler and exit")
    parser.add_argument("--version",            action="version", version=f"v{VERSION}")
    args = parser.parse_args()

    raw_workspace = _pick_workspace(args.workspace)
    workspace     = _lock_workspace(raw_workspace)

    if args.permission:    Config.PERMISSION_MODE   = args.permission
    if args.no_mcp:        Config.ENABLE_MCP        = False
    if args.no_sub_agents: Config.ENABLE_SUB_AGENTS = False

    try:
        Config.init()
    except ValueError as e:
        print(colored(f"Config error: {e}", Colors.RED))
        return 1

    print_banner()
    Log.info(f"Workspace : {workspace}")
    Log.info(f"LLM       : {Config.LLM_URL}")
    Log.info(
        f"Shell sandbox: {'workspace-only' if Config.SHELL_WORKSPACE_ONLY else 'read outside allowed (destructive still blocked)'}"
    )

    soul = SoulConfig.load(workspace)

    if args.submit:
        inbox = workspace / ".lmagent" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        fname = inbox / f"{int(time.time())}.task"
        fname.write_text(args.submit, encoding="utf-8")
        Log.success(f"Task submitted → {fname.name}")
        Log.info("Scheduler will pick it up within the next poll interval.")
        return 0

    if args.scheduler:
        run_scheduler(workspace, poll_interval=args.scheduler_interval, soul=soul)
        return 0

    if args.list_sessions:
        sessions = SessionManager(workspace).list_recent(20)
        if not sessions:
            print("No sessions found.")
            return 0
        status_colors = {
            "completed": Colors.GREEN, "waiting": Colors.MAGENTA,
            "active":    Colors.CYAN,  "error":   Colors.RED,
        }
        print(colored("\nRecent Sessions:", Colors.CYAN, bold=True))
        print(colored("─" * 80, Colors.CYAN))
        for s in sessions:
            sc = status_colors.get(s["status"], Colors.YELLOW)
            print(f"{colored(s['id'], Colors.CYAN)}  {colored(s['status'], sc)}")
            print(f"  {s['task']}")
            if s.get("parent"):
                print(colored(f"  ↳ parent: {s['parent']}", Colors.GRAY))
            print()
        return 0

    if not args.resume:
        Log.info("Checking LLM connection…")
        err = LLMClient.validate_connection()
        if err:
            Log.error(f"LLM unavailable: {err}")
            return 1
        Log.success("LLM connected\n")

    try:
        permission_mode = PermissionMode(Config.PERMISSION_MODE)
    except ValueError:
        permission_mode = PermissionMode.NORMAL

    instance_lock = InstanceLock(workspace)
    _core_instance_lock.__class__
    _core_mod._instance_lock = instance_lock

    if args.task or args.resume:
        try:
            with instance_lock:
                result = run_agent(
                    args.task or "Continue from previous session",
                    workspace,
                    permission_mode=permission_mode,
                    resume_session=args.resume,
                    plan_first=args.plan,
                    mode="interactive",
                    soul=soul,
                )
                if result.status == "waiting":
                    Log.wait(f"Session sleeping until {result.wait_until}")
                    Log.info(
                        "Auto-resume:   python agent_main.py --scheduler\n"
                        f"Manual resume: python agent_main.py --resume {result.session_id}"
                    )
                return _EXIT_CODES.get(result.status, 1)
        except RuntimeError as e:
            Log.error(str(e))
            return 1

    # ── interactive REPL + background scheduler ────────────────────────────────
    Log.info("Starting background scheduler…")
    current_session_id: Optional[str] = None

    def _current_prompt() -> str:
        if current_session_id:
            return colored(f"[{current_session_id[-8:]}]> ", Colors.YELLOW, bold=True)
        return colored("agent> ", Colors.YELLOW, bold=True)

    def _reprint_prompt() -> None:
        sys.stdout.write("\n" + _current_prompt())
        sys.stdout.flush()

    _repl_prompt_fn = _reprint_prompt

    # FIX [13]: when a scheduler worker completes, update current_session_id
    # so the next REPL input resumes that session instead of starting fresh.
    def _on_scheduler_session(sid: str) -> None:
        nonlocal current_session_id
        current_session_id = sid

    def _silent_run_scheduler(*a, **kw) -> None:
        Log.set_silent(True)
        run_scheduler(*a, **kw)

    _sched_thread = threading.Thread(
        target=_silent_run_scheduler,
        args=(workspace,),
        kwargs={
            "poll_interval":    args.scheduler_interval or Config.SCHEDULER_POLL_INTERVAL,
            "soul":             soul,
            "session_callback": _on_scheduler_session,  # FIX [13]
        },
        daemon=True,
        name="scheduler",
    )
    _sched_thread.start()
    Log.info("Scheduler running.  /help for commands, 'quit' to exit.\n")

    mcp_manager = MCPManager(workspace)
    mcp_manager.load_servers()

    while True:
        try:
            task = input(_current_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not task or task.lower() in ("quit", "exit", "q"):
            break

        if task.startswith("/"):
            parts    = task.split(maxsplit=1)
            cmd_name = parts[0]
            cmd_arg  = parts[1] if len(parts) > 1 else ""
            if cmd_name in SLASH_COMMANDS:
                result_str = SLASH_COMMANDS[cmd_name].handler(
                    workspace=workspace,
                    mode=cmd_arg,
                    session_id=current_session_id,
                    mcp_manager=mcp_manager,
                    soul=soul,
                )
                if result_str == "__RESET__":
                    current_session_id = None
                    print(colored("  ✓ Session cleared — next task starts fresh.", Colors.CYAN))
                elif result_str:
                    print(result_str)
                if cmd_name == "/mode" and cmd_arg:
                    try:
                        permission_mode = PermissionMode(cmd_arg)
                    except ValueError:
                        pass
            else:
                print(colored(f"  Unknown command: {cmd_name}. Type /help.", Colors.RED))
            continue

        try:
            with instance_lock:
                result = run_agent(
                    task,
                    workspace,
                    permission_mode=permission_mode,
                    resume_session=current_session_id,
                    mode="interactive",
                    soul=soul,
                )
                current_session_id = result.session_id

                if result.status == "waiting":
                    Log.wait(
                        f"Session sleeping until {result.wait_until}.\n"
                        f"  Scheduler will auto-resume it.\n"
                        f"  Resume manually: --resume {result.session_id}"
                    )
                    current_session_id = None

        except RuntimeError as e:
            Log.error(str(e))
        print()

    mcp_manager.close_all()
    Log.info("Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
