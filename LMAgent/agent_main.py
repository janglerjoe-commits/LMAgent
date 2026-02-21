#!/usr/bin/env python3
"""
agent_main.py — Main entrypoint for LMAgent.

Contains: run_agent(), run_scheduler(), interactive REPL,
slash commands, and the CLI argument parser.

Usage:
  python agent_main.py                        # Interactive REPL + background scheduler
  python agent_main.py "task description"     # One-shot mode
  python agent_main.py --resume SESSION_ID    # Resume session
  python agent_main.py --plan "task"          # Plan mode
  python agent_main.py --list-sessions        # List sessions
  python agent_main.py --scheduler            # Scheduler daemon only
  python agent_main.py --submit "task"        # Submit task to running scheduler

BUG FIX :
  Tool calls now execute BEFORE completion is detected.
  Previously, detect_completion() ran before _process_tool_calls(), so if the
  LLM emitted TASK_COMPLETE in the same response as a tool call (common with
  thinking models), the agent exited before the tool was ever executed.
  The fix: execute tool_calls first, then run detect_completion().
"""

import argparse
import json
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
)

import agent_core as _core_mod

_repl_prompt_fn: Optional[Callable[[], None]] = None


# =============================================================================
# WORKSPACE PICKER
# =============================================================================

def _pick_workspace(args_workspace: Optional[str]) -> str:
    """Return the workspace path to use, prompting the user if needed.

    Priority:
      1. --workspace CLI flag  (explicit, always wins)
      2. WORKSPACE environment variable  (set by .env or the shell)
      3. Interactive prompt  (shown only when neither of the above is set)

    The prompt shows the default path and lets the user press Enter to accept
    it or type an alternative.  An empty input keeps the default.
    """
    # 1. Explicit CLI arg — no prompt needed.
    if args_workspace:
        return args_workspace

    # 2. Environment variable (may have been loaded from .env by agent_core).
    env_ws = None
    import os
    raw = os.environ.get("WORKSPACE", "").strip()
    if raw:
        env_ws = raw

    if env_ws:
        return env_ws

    # 3. Interactive prompt.
    default = Config.WORKSPACE          # already set from os.getenv default
    print(colored("\n┌─ Workspace Setup ──────────────────────────────────────┐", Colors.CYAN))
    print(colored(f"│  No workspace configured.                              │", Colors.CYAN))
    print(colored(f"│  Default: {default:<45}│", Colors.CYAN))                
    print(colored("└────────────────────────────────────────────────────────┘", Colors.CYAN))
    try:
        choice = input(
            colored("  Press Enter to use the default, or type a path: ", Colors.YELLOW)
        ).strip()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment (pipe, CI, etc.) — fall through to default.
        print()
        choice = ""

    chosen = choice if choice else default

    # Offer to remember the choice in a .env file next to this script.
    if chosen != default:
        script_dir = Path(__file__).parent
        env_file   = script_dir / ".env"
        try:
            save = input(
                colored(f"  Save this path to {env_file.name} for future runs? [y/N]: ",
                        Colors.YELLOW)
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            save = ""
        if save in ("y", "yes"):
            try:
                lines: List[str] = []
                if env_file.exists():
                    lines = env_file.read_text(encoding="utf-8").splitlines()
                # Replace existing WORKSPACE= line or append a new one.
                found = False
                for i, line in enumerate(lines):
                    if line.strip().startswith("WORKSPACE="):
                        lines[i] = f'WORKSPACE="{chosen}"'
                        found = True
                        break
                if not found:
                    lines.append(f'WORKSPACE="{chosen}"')
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
    _hold_lock: bool = True,
    soul: str = "",
) -> AgentResult:

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

    # ── event emitter ─────────────────────────────────────────────────────────
    def emit(event_type: str, data: Dict[str, Any]):
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
        elif event_type == "complete":
            Log.success(f"Done — {data.get('reason', '')}")
        elif event_type == "iteration":
            Log.info(f"\nIteration {data.get('n')}/{Config.MAX_ITERATIONS}")
        elif event_type == "waiting":
            Log.wait(f"Sleeping until {data.get('resume_after')}: {data.get('reason')}")

    session_mgr = SessionManager(workspace)
    state_mgr   = StateManager(workspace)

    # ── session setup ──────────────────────────────────────────────────────────
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
            resume_msg = ws.context_on_resume.replace(
                "{current_time}", datetime.now().isoformat()
            )
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

    # ── fresh session ──────────────────────────────────────────────────────────
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
            messages.insert(1, {"role": "system",
                                 "content": f"PROJECT CONFIG:\n\n{project_cfg}"})

        detector  = LoopDetector()
        iteration = 0

    # ── managers ──────────────────────────────────────────────────────────────
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
    available_tools = TOOL_SCHEMAS.copy()
    mcp_tools       = mcp_manager.get_all_tools()
    if mcp_tools:
        available_tools.extend(mcp_tools)

    current_permission_mode = permission_mode

    # ── inner helpers ──────────────────────────────────────────────────────────

    def _save_state(wait_state: Optional[WaitState] = None):
        state_mgr.save(session_id, AgentState(
            session_id=session_id,
            iteration=iteration,
            loop_detector_state=detector.to_dict(),
            permission_mode=current_permission_mode.value,
            current_plan_step=plan_mgr.current_step_id,
            last_checkpoint=datetime.now().isoformat(),
            wait_state=wait_state.to_dict() if wait_state else None,
        ))

    def _save_session(status: str):
        session_mgr.save(session_id, messages, {
            "id":              session_id,
            "task":            task[:200],
            "iterations":      iteration,
            "status":          status,
            "permission_mode": current_permission_mode.value,
        })

    def _inject_context():
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

        # Keep the shared context reference in sync after any reassignment above.
        get_current_context()["messages"] = messages

    # ── main loop ──────────────────────────────────────────────────────────────
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
            response = LLMClient.call(messages, available_tools, stream_callback=_stream_cb)

            if mode == "interactive" and not _stream_cb.printed and response.get("content"):
                sys.stdout.write(colored("\nAssistant:\n\n", Colors.CYAN, bold=True))
                sys.stdout.flush()

            if "error" in response:
                emit("error", {"message": f"LLM error: {response['error']}"})
                return AgentResult(status="error",
                                   final_answer=f"LLM error: {response['error']}",
                                   events=events, session_id=session_id, iterations=iteration)

            content    = response.get("content", "")
            tool_calls = response.get("tool_calls") or []

            if not content and not tool_calls:
                detector.track_empty()
                continue

            if content:
                final_answer = content

            # ── WAIT detection ─────────────────────────────────────────────────
            wait_state = detect_wait(content) if content else None
            if wait_state:
                emit("waiting", {"reason": wait_state.reason,
                                  "resume_after": wait_state.resume_after})
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

            # ── BUG FIX: execute tool calls BEFORE checking for completion ─────
            #
            # Previously the order was:
            #   1. detect_completion()   ← triggered early exit if TASK_COMPLETE in text
            #   2. append assistant msg
            #   3. _process_tool_calls() ← never reached if step 1 returned True
            #
            # Thinking models (QwQ, DeepSeek-R1, etc.) commonly emit TASK_COMPLETE
            # in their reasoning text in the same response as a tool call, so the
            # old order caused the agent to exit before the tool ever ran.
            #
            # Fixed order:
            #   1. append assistant msg
            #   2. _process_tool_calls() ← tools execute first
            #   3. detect_completion()   ← check after tools have run
            # ──────────────────────────────────────────────────────────────────

            # Append the assistant message first so the conversation stays coherent.
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content:    assistant_msg["content"]    = content
            if tool_calls: assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Execute any tool calls BEFORE checking completion.
            if tool_calls:
                _, current_permission_mode = _process_tool_calls(
                    tool_calls, workspace, available_tools, detector,
                    iteration, mcp_manager, messages, current_permission_mode, emit=emit,
                )

            # Now check for completion (tools have already run).
            is_complete, reason = detect_completion(content, bool(tool_calls))

            if is_complete and task_state_mgr.current_state:
                s = task_state_mgr.current_state
                if (s.completion_gate == "processed == total"
                        and s.processed_count != s.total_count):
                    emit("warning", {"message": (f"Completion gate not met: "
                                                  f"{s.processed_count}/{s.total_count}")})
                    messages.append({
                        "role": "user",
                        "content": (f"⚠️ Cannot complete: {s.processed_count}/{s.total_count} "
                                    f"items processed. Finish remaining items first."),
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
                return AgentResult(status="completed", final_answer=final_answer,
                                   events=events, session_id=session_id, iterations=iteration)

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
        close_shell_session()       # clean up this thread's shell process
        if mode == "output":
            Log.set_silent(False)


# =============================================================================
# SCHEDULER
# =============================================================================

_running_lock: threading.Lock = threading.Lock()


def run_scheduler(workspace: Path, poll_interval: Optional[int] = None):
    poll_interval = poll_interval or Config.SCHEDULER_POLL_INTERVAL
    session_mgr   = SessionManager(workspace)
    state_mgr     = StateManager(workspace)
    inbox         = workspace / ".lmagent" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    soul          = SoulConfig.load(workspace)

    Log.info(f"Scheduler live — polling every {poll_interval}s")
    Log.info(f"Workspace : {workspace}")
    Log.info(f"Inbox     : {inbox}  (drop a .task file to submit work)")

    _running: set = set()

    def _is_running(sid: str) -> bool:
        with _running_lock: return sid in _running

    def _mark_done(sid: str):
        with _running_lock: _running.discard(sid)

    def _run_in_thread(label: str, sid: Optional[str], task_text: str):
        if sid:
            with _running_lock:
                if sid in _running: return
                _running.add(sid)

        def _worker():
            try:
                result = run_agent(
                    task=task_text,
                    workspace=workspace,
                    permission_mode=PermissionMode.AUTO,
                    resume_session=sid,
                    mode="interactive",
                    _hold_lock=False,
                    soul=soul,
                )
                Log.info(f"[{label}] finished — {result.status}")
                if result.status == "waiting":
                    Log.wait(f"[{label}] back to sleep until {result.wait_until}")
            except Exception as e:
                Log.error(f"[{label}] agent error: {e}")
            finally:
                if sid: _mark_done(sid)
                if _repl_prompt_fn is not None:
                    try:
                        _repl_prompt_fn()
                    except Exception:
                        pass

        threading.Thread(target=_worker, name=f"agent-{label}", daemon=True).start()

    try:
        while True:
            try:
                # ── inbox: new one-shot tasks ─────────────────────────────────
                for task_file in sorted(inbox.glob("*.task")):
                    try:
                        task_text = task_file.read_text(encoding="utf-8").strip()
                        if not task_text:
                            task_file.unlink(); continue
                        label = task_file.stem[:20]
                        Log.info(f"[inbox] {label} — {task_text[:60]}")
                        task_file.unlink()
                        _run_in_thread(label, None, task_text)
                    except Exception as e:
                        Log.error(f"[inbox] failed to read {task_file.name}: {e}")
                        try:
                            task_file.rename(task_file.with_suffix(".failed"))
                        except Exception:
                            pass

                # ── wake sleeping sessions ────────────────────────────────────
                next_wake_secs: Optional[float] = None

                for session in session_mgr.list_recent(200):
                    if session.get("status") != "waiting": continue
                    sid = session["id"]
                    if _is_running(sid): continue

                    saved_state = state_mgr.load(sid)

                    if not saved_state or not saved_state.wait_state:
                        Log.warning(
                            f"[{sid[-8:]}] stuck in 'waiting' with no wait_state — rescuing"
                        )
                        try:
                            data = session_mgr.load(sid)
                            if data:
                                msgs, meta = data
                                meta["status"] = "idle"
                                session_mgr.save(sid, msgs, meta)
                        except Exception as rescue_err:
                            Log.error(f"[{sid[-8:]}] rescue failed: {rescue_err}")
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
                                f"[{sid[-8:]}] sleeping — {max(0, remaining):.0f}s left  "
                                f"({ws.reason[:50]})"
                            )
                        except Exception:
                            pass
                        continue

                    Log.info(f"[{sid[-8:]}] waking — {ws.reason}")
                    _run_in_thread(sid[-8:], sid, "Continue from scheduled wake-up")

            except Exception as e:
                Log.error(f"Scheduler poll error: {e}")

            sleep_for = (
                max(0.5, next_wake_secs + 0.25)
                if next_wake_secs is not None and next_wake_secs < poll_interval
                else poll_interval
            )
            if sleep_for < poll_interval:
                Log.info(f"Scheduler: sleeping {sleep_for:.1f}s "
                         f"(next wake-up in ~{next_wake_secs:.0f}s)")
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        Log.info("Scheduler stopping — waiting for active agents…")
        deadline = time.time() + 10
        while True:
            with _running_lock:
                if not _running: break
            if time.time() > deadline: break
            time.sleep(0.5)
        Log.info("Scheduler exited.")


# =============================================================================
# BANNER
# =============================================================================

def print_banner():
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
            "  next to this script containing:  WORKSPACE=\"/your/path\"\n"
        ),
    )
    parser.add_argument("task",                 nargs="?",  help="Task to execute")
    parser.add_argument("--workspace",                      help="Workspace directory")
    parser.add_argument("--resume",             metavar="SESSION_ID",
                        help="Resume a previous session")
    parser.add_argument("--plan",               action="store_true",
                        help="Create an execution plan before running")
    parser.add_argument("--list-sessions",      action="store_true",
                        help="List recent sessions")
    parser.add_argument("--permission",         choices=["manual", "normal", "auto"],
                        help="Permission mode")
    parser.add_argument("--no-mcp",             action="store_true",
                        help="Disable MCP servers")
    parser.add_argument("--no-sub-agents",      action="store_true",
                        help="Disable sub-agents")
    parser.add_argument("--scheduler",          action="store_true",
                        help="Run as a background scheduler daemon")
    parser.add_argument("--scheduler-interval", type=int, default=None, metavar="SECONDS",
                        help=f"Scheduler poll interval (default {Config.SCHEDULER_POLL_INTERVAL}s)")
    parser.add_argument("--submit",             metavar="TASK",
                        help="Submit a task to a running scheduler and exit")
    parser.add_argument("--version",            action="version", version=f"v{VERSION}")
    args = parser.parse_args()

    # ── workspace resolution (before Config.init) ──────────────────────────────
    # _pick_workspace() handles the interactive prompt when needed and always
    # returns a concrete path string.
    chosen_workspace = _pick_workspace(args.workspace)
    Config.WORKSPACE = chosen_workspace

    if args.permission:    Config.PERMISSION_MODE   = args.permission
    if args.no_mcp:        Config.ENABLE_MCP        = False
    if args.no_sub_agents: Config.ENABLE_SUB_AGENTS = False

    try:
        Config.init()
    except ValueError as e:
        print(colored(f"Config error: {e}", Colors.RED))
        return 1

    print_banner()
    workspace = Path(Config.WORKSPACE).resolve()
    Log.info(f"Workspace : {workspace}")
    Log.info(f"LLM       : {Config.LLM_URL}")

    soul = SoulConfig.load(workspace)

    # ── --submit ───────────────────────────────────────────────────────────────
    if args.submit:
        inbox = workspace / ".lmagent" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        fname = inbox / f"{int(time.time())}.task"
        fname.write_text(args.submit, encoding="utf-8")
        Log.success(f"Task submitted → {fname.name}")
        Log.info("Scheduler will pick it up within the next poll interval.")
        return 0

    # ── --scheduler (daemon only) ──────────────────────────────────────────────
    if args.scheduler:
        run_scheduler(workspace, poll_interval=args.scheduler_interval)
        return 0

    # ── --list-sessions ────────────────────────────────────────────────────────
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

    # ── validate LLM ───────────────────────────────────────────────────────────
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
    _core_mod._instance_lock = instance_lock

    # ── one-shot mode ──────────────────────────────────────────────────────────
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

    def _reprint_prompt():
        sys.stdout.write("\n" + _current_prompt())
        sys.stdout.flush()

    _repl_prompt_fn = _reprint_prompt

    def _silent_run_scheduler(*args, **kwargs):
        # Thread-local — only silences this thread, not the REPL thread.
        Log.set_silent(True)
        run_scheduler(*args, **kwargs)

    _sched_thread = threading.Thread(
        target=_silent_run_scheduler,
        args=(workspace,),
        kwargs={"poll_interval": args.scheduler_interval or Config.SCHEDULER_POLL_INTERVAL},
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
        except (KeyboardInterrupt, EOFError):
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
