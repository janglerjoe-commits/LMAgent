#!/usr/bin/env python3
"""
LMAgent Web — v6.5.2
------------------------------------
  agent_core.py   — Config, logging, session/state/todo/plan managers, etc.
  agent_tools.py  — Tool handlers, LLMClient, _HeaderStreamCb, TOOL_SCHEMAS
  agent_main.py   — run_agent()

Place this file next to those three files.

Changes vs v6.5.1:
  - FIX (critical): Stream.flush() now uses textContent instead of MD.render()
    during live token streaming — eliminates "Page Unresponsive" freeze caused
    by the O(n²) Markdown parser running on the full accumulated buffer every
    35ms.  Full MD rendering is deferred to finalize() and wrapped in
    requestAnimationFrame() so the browser never blocks.
  - FIX: Stream.reset() now properly finalises buffered content instead of
    silently discarding it (was losing the last partial message on Stop).
  - FIX: Replay token batching wrapped in rAF chunks so replaying a long
    session history no longer freezes the page.
  - FIX: flush timer raised 35ms → 60ms to further reduce parse pressure.
  - FIX: MD.render() guarded with a 400 kB soft cap — oversized text is shown
    as a preformatted block to prevent any single render from hanging.
  - FIX: tool-row regex dotAll flag replaced with [\s\S] for older engines.
  - MINOR: onDone "done ✓" label de-duplicated (was showing twice in some
    paths).

Previous changes (v6.5.1):
  - FIX: onDone() appends a visible "✓ done" chat message.
  - FIX: SSE reconnect handler detects stuck running/waiting state.
  - FIX: Mojibake sanitization for UTF-8 special chars.
  - FIX: Agent bubble uses readable sans-serif prose; code stays monospace.

Cross-platform changes (v6.5-cross):
  - Config.WORKSPACE resolution deferred to after Config.init().
  - _TOOL_CATEGORIES System list includes "shell" alongside "powershell".
  - No direct shell calls in this file.
"""

import json
import queue
import socket
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request

# ── Import the agent — modular 3-file layout ─────────────────────────────────
try:
    import agent_core
    import agent_tools
    import agent_main as _agent_main_mod

    from agent_core import (
        Config,
        MCPManager,
        PermissionMode,
        PlanManager,
        SessionManager,
        StateManager,
        WaitState,
        SoulConfig,
        TodoManager,
        Log,
    )
    from agent_tools import (
        TOOL_SCHEMAS,
        _HeaderStreamCb as _OrigHSC,
    )
    from agent_main import run_agent

except ImportError as _ie:
    sys.exit(
        "ERROR: agent_core.py / agent_tools.py / agent_main.py not found.\n"
        f"Detail: {_ie}\n"
        "Place agent_web.py next to those three files and try again."
    )
except Exception:
    sys.exit(f"ERROR importing agent modules:\n{traceback.format_exc()}")

try:
    Config.init()
except Exception as _e:
    sys.exit(f"ERROR: Config.init() failed: {_e}")

WORKSPACE = Path(Config.WORKSPACE).resolve()
soul      = SoulConfig.load(WORKSPACE)
Log.set_silent(True)

# ── Global MCP manager ────────────────────────────────────────────────────────
_global_mcp = MCPManager(WORKSPACE)
try:
    _global_mcp.load_servers()
except Exception:
    pass

app = Flask(__name__)

_AGENT_LOCK         = threading.Lock()
_AGENT_LOCK_TIMEOUT = 300

_tl = threading.local()

_session_lock            = threading.Lock()
_current_session_id: "str | None" = None
_current_permission_mode = Config.PERMISSION_MODE

_stop_events:      "dict[str, threading.Event]" = {}
_stop_events_lock = threading.Lock()

_agent_state      = "idle"
_agent_state_lock = threading.Lock()


def _set_agent_state(s: str) -> None:
    global _agent_state
    with _agent_state_lock:
        _agent_state = s


def _get_agent_state() -> str:
    with _agent_state_lock:
        return _agent_state


# =============================================================================
# MOJIBAKE FIX  — UTF-8 bytes that were decoded as Latin-1
# =============================================================================

def _fix_mojibake(s: str) -> str:
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


# =============================================================================
# BROADCAST / STREAM QUEUES
# =============================================================================

_stream_queues: "list[queue.Queue]" = []
_stream_queues_lock = threading.Lock()


def _broadcast(item: tuple) -> None:
    kind, payload = item
    if kind in ("status", "tool", "error", "done") and isinstance(payload, str):
        payload = _fix_mojibake(payload)
    item = (kind, payload)
    _chatlog_append(item)
    with _stream_queues_lock:
        for q in _stream_queues:
            try:
                q.put_nowait(item)
            except Exception:
                pass


def _register_stream_q() -> "queue.Queue":
    q: queue.Queue = queue.Queue()
    with _stream_queues_lock:
        _stream_queues.append(q)
    return q


def _unregister_stream_q(q: "queue.Queue") -> None:
    with _stream_queues_lock:
        try:
            _stream_queues.remove(q)
        except ValueError:
            pass


# =============================================================================
# PERSISTENT CHAT LOG
# =============================================================================

_chat_logs: "dict[str, list]" = defaultdict(list)
_chat_log_lock = threading.Lock()

_REPLAY_KINDS = {"token", "tool", "status", "iteration", "done", "error", "session"}


def _chatlog_session_key() -> str:
    with _session_lock:
        return _current_session_id or "_none"


def _chatlog_append(item: tuple) -> None:
    kind, payload = item
    if kind not in _REPLAY_KINDS:
        return
    key = _chatlog_session_key()
    with _chat_log_lock:
        _chat_logs[key].append(item)


def _chatlog_get(session_id: "str | None") -> "list[tuple]":
    key = session_id or "_none"
    with _chat_log_lock:
        return list(_chat_logs.get(key, []))


def _chatlog_clear(session_id: "str | None") -> None:
    key = session_id or "_none"
    with _chat_log_lock:
        _chat_logs.pop(key, None)


def _chatlog_merge_keys(old_key: "str | None", new_key: str) -> None:
    old = old_key or "_none"
    if old == new_key:
        return
    with _chat_log_lock:
        if old in _chat_logs:
            _chat_logs[new_key] = _chat_logs.pop(old, []) + _chat_logs.get(new_key, [])


# =============================================================================
# STOP EXCEPTION
# =============================================================================

class _AgentStopped(BaseException):
    """Raised inside the agent thread when the user clicks Stop."""


# =============================================================================
# _HeaderStreamCb PATCH
# =============================================================================

class _PatchedHSC(_OrigHSC):
    def __call__(self, token: str):
        super().__call__(token)
        if token:
            cb = getattr(_tl, "token_cb", None)
            if cb:
                try:
                    cb(token)
                except _AgentStopped:
                    raise
                except Exception:
                    pass


agent_tools._HeaderStreamCb = _PatchedHSC
_agent_main_mod._HeaderStreamCb = _PatchedHSC


# =============================================================================
# THINKING FILTER
# =============================================================================

class _ThinkingFilter:
    def __init__(self):
        self._buf = ""
        self._in  = False

    def feed(self, tok: str) -> None:
        self._buf += tok
        while True:
            if self._in:
                end = self._buf.find("</think>")
                if end == -1:
                    if len(self._buf) > 100_000:
                        self._buf = ""
                    return
                self._buf = self._buf[end + len("</think>"):]
                self._in  = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    safe_len = max(0, len(self._buf) - 6)
                    if safe_len:
                        _broadcast(("token", self._buf[:safe_len]))
                        self._buf = self._buf[safe_len:]
                    return
                if start > 0:
                    _broadcast(("token", self._buf[:start]))
                self._buf = self._buf[start + len("<think>"):]
                self._in  = True

    def flush(self) -> None:
        if not self._in and self._buf:
            _broadcast(("token", self._buf))
        self._buf = ""
        self._in  = False


# =============================================================================
# EVENT NOISE FILTER
# =============================================================================

_STATUS_NOISE = frozenset({
    "thinking model mode", "state restored", "soul loaded", "mcp ",
    "checking llm", "llm connected", "task:", "workspace :", "llm       :",
    "loaded ", "compacting context", "compacted:", "plan approved", "plan mode",
})


def _is_noisy(msg: str) -> bool:
    ml = msg.lower()
    return any(ml.startswith(kw) or kw in ml[:40] for kw in _STATUS_NOISE)


# =============================================================================
# SHARED EVENT HANDLER
# =============================================================================

def _push_event(event, stop_event: "threading.Event | None", last_status: list,
                filt: "_ThinkingFilter | None" = None) -> None:
    if stop_event and stop_event.is_set():
        return
    etype = event.type
    edata = event.data
    if etype == "tool_call":
        if filt:
            filt.flush()
        name    = edata.get("name", "?")
        preview = edata.get("args_preview", "")[:50]
        _broadcast(("tool", f"{name}({preview})"))
    elif etype == "tool_result":
        mark = "✓" if edata.get("success") else "✗"
        _broadcast(("tool", f"{mark} {edata.get('name', '')}"))
    elif etype == "iteration":
        if filt:
            filt.flush()
        _broadcast(("iteration", f"{edata.get('n')}/{edata.get('max')}"))
    elif etype in ("log", "warning", "error"):
        msg = edata.get("message") or edata.get("error") or ""
        if not msg:
            return
        if etype == "error" or (not _is_noisy(msg) and msg != last_status[0]):
            last_status[0] = msg
            _broadcast(("status", msg[:100]))
    elif etype == "waiting":
        _broadcast(("status", f"waiting until {edata.get('resume_after')}"))
    elif etype == "complete":
        _broadcast(("status", f"done — {edata.get('reason', '')}"))


def _sse_response(generator) -> Response:
    return Response(
        generator,
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "Content-Type":      "text/event-stream; charset=utf-8",
        },
    )


# =============================================================================
# TOOL CATEGORIES
# =============================================================================

_TOOL_CATEGORIES = {
    "Files":  ["read", "write", "edit", "glob", "grep", "ls", "mkdir"],
    "Git":    ["git_status", "git_diff", "git_add", "git_commit", "git_branch"],
    "System": ["shell", "powershell", "get_time"],
    "Tasks":  ["todo_add", "todo_complete", "todo_update", "todo_list",
               "plan_complete_step", "task"],
    "State":  ["task_state_update", "task_state_get", "task_reconcile"],
}


def _build_tools_payload() -> dict:
    builtin: dict = {}
    for cat, names in _TOOL_CATEGORIES.items():
        tools = []
        for schema in TOOL_SCHEMAS:
            fn = schema.get("function", {})
            if fn.get("name") in names:
                tools.append({
                    "name":        fn["name"],
                    "description": fn.get("description", ""),
                    "params":      list(fn.get("parameters", {}).get("properties", {}).keys()),
                    "required":    fn.get("parameters", {}).get("required", []),
                })
        if tools:
            builtin[cat] = tools

    mcp_groups: dict = {}
    for client in _global_mcp.clients:
        if not client.health_check():
            continue
        tools = []
        for t in client.tools:
            tools.append({
                "name":        t.get("name", ""),
                "description": t.get("description", ""),
                "params":      list(t.get("inputSchema", {}).get("properties", {}).keys()),
                "required":    t.get("inputSchema", {}).get("required", []),
            })
        if tools:
            mcp_groups[f"MCP · {client.name}"] = tools

    return {"builtin": builtin, "mcp": mcp_groups}


# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>LMAgent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0d0d0f;
  --bg2:       #121214;
  --surface:   #16161a;
  --surface2:  #1c1c22;
  --border:    #252530;
  --border2:   #32323e;
  --amber:     #e8a245;
  --amber-dim: #7a4e18;
  --teal:      #3ecfb2;
  --teal-dim:  #0e4037;
  --red:       #e05555;
  --green:     #3ec97a;
  --purple:    #9d7dea;
  --text:      #ddd8d0;
  --text2:     #8e8c88;
  --text3:     #4e4c50;
  --user-bg:   #14131f;
  --user-bdr:  #27254a;
  --agent-bg:  #111113;
  --agent-bdr: #1f1f25;
  --code-bg:   #09090b;
  --code-bdr:  #1a1a20;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
  --r: 8px;
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13.5px;
  line-height: 1.6;
  overflow: hidden;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
::-webkit-scrollbar-track { background: transparent; }

#app {
  display: flex;
  flex-direction: column;
  height: 100dvh;
  max-width: 860px;
  margin: 0 auto;
  position: relative;
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
  z-index: 5;
  gap: 10px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 9px;
  font-family: var(--sans);
  font-weight: 700;
  font-size: 15px;
  color: var(--text);
  letter-spacing: -0.2px;
}

.logo-mark {
  width: 28px; height: 28px;
  background: var(--amber);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; color: #0d0d0f; font-weight: 700;
  flex-shrink: 0; font-family: var(--mono);
}

.logo-sub { color: var(--text2); font-weight: 400; font-size: 12px; margin-left: 2px; }
.header-right { display: flex; gap: 6px; align-items: center; }

.session-pill {
  font-size: 10px; font-family: var(--mono);
  color: var(--text3); background: var(--bg);
  border: 1px solid var(--border); border-radius: 3px;
  padding: 2px 7px; display: none; letter-spacing: 0.04em;
}

.conn-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text3); flex-shrink: 0;
  transition: background .4s;
}
.conn-dot.ok   { background: var(--green); }
.conn-dot.err  { background: var(--red); }
.conn-dot.try  { background: var(--amber); animation: pulse 1s ease-in-out infinite; }

.btn {
  font-family: var(--mono); font-size: 11px;
  padding: 5px 12px; border-radius: 5px;
  border: 1px solid var(--border); background: transparent;
  color: var(--text2); cursor: pointer;
  transition: border-color .12s, color .12s;
  white-space: nowrap; user-select: none;
}
.btn:hover { border-color: var(--amber); color: var(--amber); }
.btn.danger:hover { border-color: var(--red); color: var(--red); }
.btn.accent { border-color: var(--teal-dim); color: var(--teal); }
.btn.accent:hover { border-color: var(--teal); }

#messages {
  flex: 1; overflow-y: auto;
  padding: 16px 16px 8px;
  display: flex; flex-direction: column; gap: 0;
}

#empty {
  margin: auto; text-align: center; color: var(--text3);
  padding: 40px 20px; display: flex; flex-direction: column;
  align-items: center; gap: 12px; user-select: none;
}
.empty-glyph {
  font-size: 32px; opacity: 0.4; letter-spacing: -4px;
  font-family: var(--mono); color: var(--amber);
}
#empty p { font-size: 12px; max-width: 200px; line-height: 1.5; }
#empty .hint {
  font-size: 11px; border: 1px solid var(--border);
  border-radius: 4px; padding: 4px 10px; color: var(--text3);
}

.msg { display: flex; flex-direction: column; animation: fadeUp .14s ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(3px); } }

.msg-label {
  font-size: 10px; font-family: var(--sans); font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase;
  color: var(--text3); padding: 12px 2px 3px;
}
.msg.user  .msg-label { color: var(--amber); }
.msg.agent .msg-label { color: var(--teal); }

.msg-body {
  padding: 13px 16px; border-radius: var(--r);
  border: 1px solid var(--border);
  word-break: break-word; overflow-x: auto;
}
.msg.user  .msg-body {
  background: var(--user-bg); border-color: var(--user-bdr);
  white-space: pre-wrap; line-height: 1.65;
}

/* ── Agent bubble: readable sans-serif prose ── */
.msg.agent .msg-body {
  background: var(--agent-bg); border-color: var(--agent-bdr);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.75;
  color: var(--text);
}

/* streaming state: monospace plain text, no flicker */
.msg.agent .msg-body.streaming {
  font-family: var(--mono);
  font-size: 13px;
  white-space: pre-wrap;
  color: var(--text2);
}

.msg.agent .msg-body h1,.msg.agent .msg-body h2,
.msg.agent .msg-body h3,.msg.agent .msg-body h4 {
  font-family: var(--sans); font-weight: 600;
  color: var(--text); margin: 1em 0 .35em; line-height: 1.3;
}
.msg.agent .msg-body h1 {
  font-size: 1.2em;
  border-bottom: 1px solid var(--border2);
  padding-bottom: .3em;
}
.msg.agent .msg-body h2 { font-size: 1.05em; }
.msg.agent .msg-body h3 { font-size: 1em; color: var(--text2); }
.msg.agent .msg-body p  { margin: .5em 0; }
.msg.agent .msg-body p:first-child { margin-top: 0; }
.msg.agent .msg-body p:last-child  { margin-bottom: 0; }
.msg.agent .msg-body strong { color: var(--text); font-weight: 600; }
.msg.agent .msg-body em     { font-style: italic; color: var(--text2); }
.msg.agent .msg-body del    { text-decoration: line-through; color: var(--text3); }
.msg.agent .msg-body a {
  color: var(--teal); text-decoration: underline;
  text-underline-offset: 2px; word-break: break-all;
}
.msg.agent .msg-body a:hover { color: var(--amber); }

/* inline code stays monospace */
.msg.agent .msg-body code {
  background: var(--code-bg); border: 1px solid var(--code-bdr);
  border-radius: 3px; padding: 1px 6px;
  font-family: var(--mono); font-size: .82em; color: var(--amber);
}
/* code blocks */
.msg.agent .msg-body pre {
  background: var(--code-bg); border: 1px solid var(--code-bdr);
  border-radius: 6px; padding: 13px 15px; overflow-x: auto;
  margin: .7em 0; position: relative;
}
.msg.agent .msg-body pre code {
  background: none; border: none; padding: 0;
  font-size: .83em; color: var(--text2); white-space: pre;
  font-family: var(--mono);
}
.msg.agent .msg-body pre .lang {
  position: absolute; top: 7px; right: 9px;
  font-size: 9px; color: var(--text3);
  text-transform: uppercase; letter-spacing: .06em;
  font-family: var(--mono);
}
.msg.agent .msg-body ul,
.msg.agent .msg-body ol { padding-left: 1.5em; margin: .5em 0; }
.msg.agent .msg-body li { margin: .2em 0; }
.msg.agent .msg-body blockquote {
  border-left: 2px solid var(--teal-dim); margin: .5em 0;
  padding: .25em .8em .25em 1em; color: var(--text2);
  background: rgba(62,207,178,.04); border-radius: 0 4px 4px 0;
  font-style: italic;
}
.msg.agent .msg-body hr { border: none; border-top: 1px solid var(--border2); margin: .9em 0; }
.msg.agent .msg-body table {
  border-collapse: collapse; width: 100%;
  margin: .6em 0; font-size: .88em; font-family: var(--mono);
}
.msg.agent .msg-body th,
.msg.agent .msg-body td { border: 1px solid var(--border2); padding: 5px 10px; text-align: left; }
.msg.agent .msg-body th { background: var(--surface2); color: var(--text2); font-weight: 600; }
.msg.agent .msg-body tr:nth-child(even) td { background: rgba(255,255,255,.015); }

/* system / status messages */
.msg.sys .msg-label { display: none; }
.msg.sys .msg-body {
  background: transparent; border: none;
  border-left: 2px solid var(--border2); border-radius: 0;
  padding: 2px 10px; color: var(--text3);
  font-size: 11.5px; font-style: italic; white-space: pre-wrap;
  font-family: var(--mono);
}
.msg.sys.err  .msg-body { border-left-color: var(--red);    color: #c07070; }
.msg.sys.ok   .msg-body { border-left-color: var(--green);  color: #60b070; }
.msg.sys.warn .msg-body { border-left-color: var(--amber);  color: #b08040; }
.msg.sys.info .msg-body {
  border-left-color: var(--purple); color: var(--text2);
  font-style: normal; font-size: 12px;
}

.tool-group { display: flex; flex-direction: column; padding: 2px 0; }
.tool-row {
  display: flex; align-items: center; gap: 6px;
  padding: 1px 0 1px 12px; font-size: 11px; color: var(--text3);
  min-height: 19px; animation: fadeUp .1s ease both;
  font-family: var(--mono);
}
.tr-icon { width: 11px; flex-shrink: 0; font-size: 10px; transition: color .2s; }
.tr-name { color: var(--text2); font-weight: 500; }
.tr-args { color: var(--text3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 240px; }
.tool-row[data-s="pending"] .tr-icon { color: var(--text3); animation: spin .8s linear infinite; }
.tool-row[data-s="ok"]      .tr-icon { color: var(--green); }
.tool-row[data-s="fail"]    .tr-icon { color: var(--red); }
@keyframes spin { to { transform: rotate(360deg); } }
.tool-more {
  font-size: 10px; color: var(--text3);
  padding: 1px 0 1px 12px; cursor: pointer; transition: color .12s;
  font-family: var(--mono);
}
.tool-more:hover { color: var(--text2); }

.cursor {
  display: inline-block; width: 2px; height: 1em;
  background: var(--amber); margin-left: 1px;
  vertical-align: text-bottom; animation: blink .6s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

#scroll-btn {
  position: absolute; bottom: 90px; right: 18px;
  width: 30px; height: 30px; border-radius: 50%;
  background: var(--surface); border: 1px solid var(--border2);
  color: var(--text2); font-size: 14px; cursor: pointer;
  display: none; align-items: center; justify-content: center;
  box-shadow: 0 2px 8px rgba(0,0,0,.5);
  transition: border-color .12s, color .12s; z-index: 4;
}
#scroll-btn.show { display: flex; }
#scroll-btn:hover { border-color: var(--amber); color: var(--amber); }

#status-bar {
  display: flex; align-items: center; gap: 7px;
  padding: 4px 16px; font-size: 11px; color: var(--text3);
  border-top: 1px solid var(--border); background: var(--surface);
  flex-shrink: 0; min-height: 24px; font-family: var(--mono);
}
.dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--text3); flex-shrink: 0; transition: background .25s;
}
.dot.run  { background: var(--amber);  animation: pulse 1s ease-in-out infinite; }
.dot.wait { background: var(--purple); animation: pulse 1s ease-in-out infinite; }
.dot.done { background: var(--green); }
.dot.err  { background: var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
#status-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#iter-badge  { font-size: 10px; border: 1px solid var(--border); border-radius: 3px; padding: 1px 6px; display: none; }
#mode-badge  { font-size: 10px; border: 1px solid var(--border); border-radius: 3px; padding: 1px 6px; color: var(--text3); }

#input-area {
  display: flex; flex-direction: column;
  padding: 9px 16px 11px;
  border-top: 1px solid var(--border); background: var(--surface);
  flex-shrink: 0; gap: 6px;
}
#input-row { display: flex; gap: 8px; align-items: flex-end; }

#msg-input {
  flex: 1; background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--r); color: var(--text); font-family: var(--mono);
  font-size: 13.5px; padding: 8px 12px; resize: none; outline: none;
  max-height: 140px; overflow-y: auto; transition: border-color .12s;
  line-height: 1.5; min-height: 40px;
}
#msg-input:focus { border-color: var(--amber); }
#msg-input::placeholder { color: var(--text3); }
#msg-input:disabled { opacity: 0.5; cursor: not-allowed; }

#send-btn {
  background: var(--amber); color: #0d0d0f; border: none;
  border-radius: var(--r); padding: 8px 18px;
  font-family: var(--sans); font-weight: 700; font-size: 12px;
  cursor: pointer; transition: all .12s; flex-shrink: 0;
  height: 40px; letter-spacing: .02em;
}
#send-btn:hover { background: #f0b055; }
#send-btn.stop { background: transparent; border: 1px solid var(--red); color: var(--red); }
#send-btn.stop:hover { background: rgba(224,85,85,.1); }

#palette {
  border: 1px solid var(--border2); border-radius: 6px;
  background: var(--bg2); overflow: hidden; display: none;
}
#palette.open { display: block; }
.pi {
  display: flex; align-items: baseline; gap: 10px;
  padding: 7px 12px; cursor: pointer; transition: background .1s; font-size: 12px;
}
.pi:hover, .pi.sel { background: var(--surface2); }
.pi-cmd { color: var(--amber); font-weight: 600; min-width: 90px; }
.pi-desc { color: var(--text3); font-size: 11px; }

.panel {
  position: absolute; top: 0; right: 0;
  width: min(320px, 90vw); height: 100%;
  background: var(--surface); border-left: 1px solid var(--border);
  transform: translateX(100%); transition: transform .2s ease;
  z-index: 20; display: flex; flex-direction: column;
}
.panel.open { transform: translateX(0); }
#tools-panel { width: min(360px, 92vw); }

.panel-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 14px; border-bottom: 1px solid var(--border);
  font-family: var(--sans); font-weight: 600; font-size: 12px;
  color: var(--text2); flex-shrink: 0;
}

#sessions-list { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 4px; }
.session-item {
  padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border);
  cursor: pointer; transition: all .12s; background: var(--bg);
}
.session-item:hover { border-color: var(--teal-dim); background: var(--surface2); }
.session-item.active { border-color: var(--amber); }
.si-id   { font-size: 10px; color: var(--text3); margin-bottom: 2px; }
.si-task { font-size: 11.5px; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.si-stat { font-size: 10px; margin-top: 2px; color: var(--text3); }
.si-stat.completed { color: var(--green); }
.si-stat.error, .si-stat.interrupted { color: var(--red); }
.si-stat.waiting { color: var(--purple); }
.si-stat.active  { color: var(--amber); }

#tools-search {
  margin: 8px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 5px; color: var(--text); font-family: var(--mono);
  font-size: 12px; padding: 6px 10px; outline: none;
  width: calc(100% - 16px); flex-shrink: 0;
}
#tools-search:focus { border-color: var(--teal); }
#tools-list { flex: 1; overflow-y: auto; padding: 4px 8px 12px; }

.tool-cat { margin-bottom: 6px; }
.tool-cat-hdr {
  font-family: var(--sans); font-weight: 600; font-size: 10px;
  letter-spacing: .09em; text-transform: uppercase;
  color: var(--text3); padding: 7px 4px 3px;
}
.mcp-cat .tool-cat-hdr { color: var(--teal); }
.tool-entry {
  padding: 6px 8px; border-radius: 5px; border: 1px solid transparent;
  margin-bottom: 1px; transition: all .1s;
}
.tool-entry:hover { background: var(--surface2); border-color: var(--border); }
.te-name { font-size: 12px; color: var(--text); font-weight: 500; }
.te-params { color: var(--text3); font-size: 11px; font-weight: 400; }
.te-desc { font-size: 11px; color: var(--text2); margin-top: 1px; line-height: 1.4; font-family: var(--sans); }

#overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.3); z-index: 15;
}
#overlay.show { display: block; }

/* Replay banner */
#replay-banner {
  display: none; text-align: center;
  font-size: 11px; color: var(--text3);
  padding: 4px 0 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}
#replay-banner.show { display: block; }

@media (max-width: 520px) {
  header, #input-area, #messages { padding-left: 11px; padding-right: 11px; }
}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo">
      <div class="logo-mark">&#955;</div>
      LMAgent
      <span class="logo-sub">v6.5.2</span>
    </div>
    <div class="header-right">
      <span class="session-pill" id="session-pill"></span>
      <div class="conn-dot" id="conn-dot" title="Stream connection"></div>
      <button class="btn accent" onclick="UI.toggleTools()">Tools</button>
      <button class="btn" onclick="UI.toggleSessions()">Sessions</button>
      <button class="btn danger" onclick="Agent.newSession()">New</button>
    </div>
  </header>

  <div id="messages">
    <div id="replay-banner"></div>
    <div id="empty">
      <div class="empty-glyph">&#955;_</div>
      <p>Send a message to start</p>
      <div class="hint">/ for commands</div>
    </div>
  </div>

  <button id="scroll-btn" onclick="Scroll.jump()" title="&#8595;">&#8595;</button>

  <div id="status-bar">
    <div class="dot" id="dot"></div>
    <span id="status-text">ready</span>
    <span id="iter-badge"></span>
    <span id="mode-badge">auto</span>
  </div>

  <div id="input-area">
    <div id="palette"></div>
    <div id="input-row">
      <textarea id="msg-input" rows="1" placeholder="Message or /command&hellip;"
        onkeydown="Palette.onKey(event)"
        oninput="Palette.onInput(this)"></textarea>
      <button id="send-btn" onclick="Agent.sendOrStop()">Send</button>
    </div>
  </div>
</div>

<div id="overlay" onclick="UI.closeAll()"></div>

<div class="panel" id="sessions-panel">
  <div class="panel-hdr">
    <span>Sessions</span>
    <button class="btn" onclick="UI.closeSessions()">&#x2715;</button>
  </div>
  <div id="sessions-list"></div>
</div>

<div class="panel" id="tools-panel">
  <div class="panel-hdr">
    <span>Tools</span>
    <button class="btn" onclick="UI.closeTools()">&#x2715;</button>
  </div>
  <input id="tools-search" placeholder="Filter&hellip;" oninput="ToolsPanel.filter(this.value)">
  <div id="tools-list"></div>
</div>

<script>
'use strict';

// ═══════════════════════════════════════════════════════════════
// UNICODE SANITIZER  — fix mojibake before rendering
// ═══════════════════════════════════════════════════════════════
const _TD = new TextDecoder('utf-8', { fatal: true });
function fixMojibake(s) {
  if (!s) return s;
  return s.replace(
    /[Â-ß][\x80-\xBF]|[à-ï][\x80-\xBF]{2}|[ð-÷][\x80-\xBF]{3}/g,
    function(match) {
      try {
        var b = new Uint8Array(match.length);
        for (var i = 0; i < match.length; i++) b[i] = match.charCodeAt(i) & 0xff;
        return _TD.decode(b);
      } catch (_) { return match; }
    }
  );
}


// ═══════════════════════════════════════════════════════════════
// MARKDOWN — safe render with size guard
// ═══════════════════════════════════════════════════════════════
const MD = (() => {
  const MAX_RENDER_BYTES = 400000; // ~400 kB — beyond this we show as pre
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  function inline(s) {
    s = s.replace(/`([^`]+)`/g, function(_,c){ return '<code>'+esc(c)+'</code>'; });
    s = s.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g,     '<strong>$1</strong>');
    s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
    s = s.replace(/_(.+?)_/g,   '<em>$1</em>');
    s = s.replace(/~~(.+?)~~/g, '<del>$1</del>');
    s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(_,a,src){ return '<img src="'+esc(src)+'" alt="'+esc(a)+'" style="max-width:100%">'; });
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,  function(_,t,href){ return '<a href="'+esc(href)+'" target="_blank" rel="noopener">'+t+'</a>'; });
    s = s.replace(/(?<!["'>])(https?:\/\/[^\s<>"]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
    return s;
  }

  function parse(md) {
    var lines = md.split('\n');
    var out   = [];
    var i = 0;
    var peek = function(){ return lines[i]; };
    var take = function(){ return lines[i++]; };

    function listBlock(tag) {
      var items = [];
      var base  = ((peek().match(/^(\s*)/) || ['',''])[1]).length;
      while (i < lines.length) {
        var l = peek();
        if (/^\s*$/.test(l)) { take(); break; }
        var ind = ((l.match(/^(\s*)/) || ['',''])[1]).length;
        if (ind < base && items.length) break;
        if (!/^(\s*)([-*+]|\d+\.)\s/.test(l)) break;
        var content = take().replace(/^\s*([-*+]|\d+\.)\s+/, '');
        var nested = '';
        if (i < lines.length) {
          var ni = ((lines[i].match(/^(\s*)/) || ['',''])[1]).length;
          if (ni > ind && /^(\s*)([-*+]|\d+\.)\s/.test(lines[i]))
            nested = listBlock(/^(\s*)([-*+])\s/.test(lines[i]) ? 'ul' : 'ol');
        }
        items.push('<li>'+inline(content)+nested+'</li>');
      }
      return '<'+tag+'>'+items.join('')+'</'+tag+'>';
    }

    function tableBlock() {
      var hdr  = take(); take();
      var headers = hdr.split('|').map(function(c){ return c.trim(); }).filter(Boolean);
      var rows = [];
      while (i < lines.length && /\|/.test(peek()) && !/^\s*$/.test(peek()))
        rows.push(take().split('|').map(function(c){ return c.trim(); }).filter(Boolean));
      var ths = headers.map(function(h){ return '<th>'+inline(h)+'</th>'; }).join('');
      var trs = rows.map(function(r){ return '<tr>'+r.map(function(c){ return '<td>'+inline(c)+'</td>'; }).join('')+'</tr>'; }).join('');
      return '<table><thead><tr>'+ths+'</tr></thead><tbody>'+trs+'</tbody></table>';
    }

    while (i < lines.length) {
      var line = peek();
      if (/^```/.test(line)) {
        var lang = line.slice(3).trim(); take();
        var code = [];
        while (i < lines.length && !/^```/.test(peek())) code.push(esc(take()));
        if (i < lines.length) take();
        out.push('<pre>'+(lang ? '<span class="lang">'+esc(lang)+'</span>' : '')+'<code>'+code.join('\n')+'</code></pre>');
        continue;
      }
      if (/^(\*{3,}|-{3,}|_{3,})\s*$/.test(line)) { take(); out.push('<hr>'); continue; }
      var hm = line.match(/^(#{1,6})\s+(.+)/);
      if (hm) { take(); out.push('<h'+hm[1].length+'>'+inline(hm[2])+'</h'+hm[1].length+'>'); continue; }
      if (/^>\s?/.test(line)) {
        var q = [];
        while (i < lines.length && /^>\s?/.test(peek())) q.push(take().replace(/^>\s?/, ''));
        out.push('<blockquote>'+parse(q.join('\n'))+'</blockquote>');
        continue;
      }
      if (/^(\s*)([-*+])\s/.test(line)) { out.push(listBlock('ul')); continue; }
      if (/^(\s*)\d+\.\s/.test(line))   { out.push(listBlock('ol')); continue; }
      if (/\|/.test(line) && i+1 < lines.length && /^\|?[\s:|-]+\|/.test(lines[i+1])) {
        out.push(tableBlock()); continue;
      }
      if (/^\s*$/.test(line)) { take(); continue; }
      var para = [];
      while (i < lines.length) {
        var ll = peek();
        if (/^\s*$/.test(ll)) break;
        if (/^(#{1,6}\s|```|>\s?|[-*+]\s|\d+\.\s|(\*{3,}|-{3,}|_{3,})\s*$)/.test(ll)) break;
        para.push(take());
      }
      if (para.length) out.push('<p>'+inline(para.join(' '))+'</p>');
    }
    return out.join('\n');
  }

  function render(md) {
    if (!md || !md.trim()) return '';
    // Guard: if text is enormous, skip expensive parse and show as preformatted
    if (md.length > MAX_RENDER_BYTES) {
      return '<pre style="white-space:pre-wrap;word-break:break-word">' + esc(md) + '</pre>';
    }
    return parse(md);
  }

  return { render };
})();


// ═══════════════════════════════════════════════════════════════
// SCROLL
// ═══════════════════════════════════════════════════════════════
const Scroll = (() => {
  const pane = function(){ return document.getElementById('messages'); };
  const btn  = function(){ return document.getElementById('scroll-btn'); };
  var pinned = true;

  function atBottom() {
    var el = pane();
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }
  function update() { btn().classList.toggle('show', !pinned && Agent.running()); }
  function maybe()  { if (pinned) pane().scrollTop = pane().scrollHeight; }
  function jump()   { pinned = true; pane().scrollTop = pane().scrollHeight; update(); }
  function onScroll() {
    if (atBottom()) pinned = true;
    else if (Agent.running()) pinned = false;
    update();
  }
  pane().addEventListener('scroll', onScroll, { passive: true });
  return { maybe, jump, pin: function(){ pinned = true; }, update };
})();


// ═══════════════════════════════════════════════════════════════
// STATUS BAR
// ═══════════════════════════════════════════════════════════════
const Status = (() => {
  var dot  = function(){ return document.getElementById('dot'); };
  var text = function(){ return document.getElementById('status-text'); };
  var iter = function(){ return document.getElementById('iter-badge'); };
  var mode = function(){ return document.getElementById('mode-badge'); };
  return {
    set: function(msg, state) {
      state = state || '';
      text().textContent = msg || '';
      dot().className = 'dot ' + state;
    },
    iter: function(v) {
      var el = iter();
      el.textContent = v || '';
      el.style.display = v ? '' : 'none';
    },
    mode: function(v) { mode().textContent = v || 'auto'; },
  };
})();


// ═══════════════════════════════════════════════════════════════
// CONNECTION DOT
// ═══════════════════════════════════════════════════════════════
const ConnDot = (() => {
  var el = function(){ return document.getElementById('conn-dot'); };
  return {
    ok:  function(){ el().className = 'conn-dot ok';  el().title = 'Stream connected'; },
    err: function(){ el().className = 'conn-dot err'; el().title = 'Stream disconnected'; },
    try: function(){ el().className = 'conn-dot try'; el().title = 'Reconnecting\u2026'; },
  };
})();


// ═══════════════════════════════════════════════════════════════
// CHAT LOG REPLAY  — chunked to avoid blocking
// ═══════════════════════════════════════════════════════════════
const Replay = (() => {
  function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  function _applyEvents(events) {
    var tokenBuf       = '';
    var toolGroupItems = [];
    var pane           = document.getElementById('messages');

    function flushTokens() {
      if (!tokenBuf.trim()) { tokenBuf = ''; return; }
      document.getElementById('empty') && document.getElementById('empty').remove();
      var wrap = document.createElement('div');
      wrap.className = 'msg agent';
      wrap.innerHTML = '<div class="msg-label">Agent</div><div class="msg-body"></div>';
      var body = wrap.querySelector('.msg-body');
      var fixed = fixMojibake(tokenBuf);
      body.innerHTML = MD.render(fixed) || _esc(fixed);
      pane.appendChild(wrap);
      tokenBuf = '';
    }

    function flushTools() {
      if (!toolGroupItems.length) return;
      document.getElementById('empty') && document.getElementById('empty').remove();
      var group = document.createElement('div');
      group.className = 'tool-group';
      for (var ti = 0; ti < toolGroupItems.length; ti++) {
        var t   = toolGroupItems[ti];
        var row = document.createElement('div');
        row.className = 'tool-row';
        var ok  = t.ok;
        row.dataset.s = ok === null ? 'pending' : (ok ? 'ok' : 'fail');
        var icon    = ok === null ? '\u25cc' : (ok ? '\u2713' : '\u2717');
        var preview = t.args ? (t.args.slice(0,50) + (t.args.length > 50 ? '\u2026' : '')) : '';
        row.innerHTML = '<span class="tr-icon">'+icon+'</span><span class="tr-name">'+_esc(t.name)+'</span>'+(preview ? '<span class="tr-args">('+_esc(preview)+')</span>' : '');
        group.appendChild(row);
      }
      pane.appendChild(group);
      toolGroupItems = [];
    }

    var lastWasToken = false;

    for (var ei = 0; ei < events.length; ei++) {
      var kind    = events[ei][0];
      var payload = events[ei][1];

      if (kind === 'token') {
        if (!lastWasToken) { flushTools(); }
        tokenBuf += payload;
        lastWasToken = true;
      } else if (kind === 'tool') {
        if (lastWasToken) { flushTokens(); }
        lastWasToken = false;
        var t2 = payload;
        if (t2.startsWith('\u2713') || t2.startsWith('\u2717')) {
          var name2 = t2.slice(1).trim();
          for (var ri = toolGroupItems.length - 1; ri >= 0; ri--) {
            if (toolGroupItems[ri].name === name2 && toolGroupItems[ri].ok === null) {
              toolGroupItems[ri].ok = t2.startsWith('\u2713');
              break;
            }
          }
        } else {
          flushTokens();
          var m2   = t2.match(/^([^(]+)\(?([\s\S]*?)\)?$/);
          var n2   = m2 ? m2[1].trim() : t2;
          var a2   = m2 ? m2[2].trim() : '';
          toolGroupItems.push({ name: n2, args: a2, ok: null });
        }
      } else if (kind === 'iteration') {
        flushTokens(); flushTools(); lastWasToken = false;
        var mm = String(payload).match(/(\d+)\/(\d+)/);
        if (mm) _sysMsg('\u2014 iteration '+mm[1]+'/'+mm[2]+' \u2014', 'info');
      } else if (kind === 'done') {
        flushTokens(); flushTools(); lastWasToken = false;
      } else if (kind === 'error') {
        flushTokens(); flushTools(); lastWasToken = false;
        _sysMsg('\u26a0 ' + payload, 'err');
      }
    }
    flushTokens();
    flushTools();
  }

  function _sysMsg(text, variant) {
    var d = document.createElement('div');
    d.className = 'msg sys ' + (variant || '');
    d.innerHTML = '<div class="msg-label"></div><div class="msg-body"></div>';
    d.querySelector('.msg-body').textContent = text;
    document.getElementById('messages').appendChild(d);
  }

  function run(events) {
    if (!events || !events.length) return;
    var banner = document.getElementById('replay-banner');
    banner.textContent = '\u21a9 restored ' + events.length + ' events from this session';
    banner.classList.add('show');
    setTimeout(function(){ banner.classList.remove('show'); }, 3000);

    // Chunk replay over multiple rAF frames to avoid blocking
    var CHUNK = 200;
    var offset = 0;
    function nextChunk() {
      var slice = events.slice(offset, offset + CHUNK);
      if (!slice.length) { Scroll.jump(); return; }
      _applyEvents(slice);
      offset += CHUNK;
      requestAnimationFrame(nextChunk);
    }
    requestAnimationFrame(nextChunk);
  }

  return { run };
})();


// ═══════════════════════════════════════════════════════════════
// MESSAGES
// ═══════════════════════════════════════════════════════════════
const Messages = (() => {
  var pane = function(){ return document.getElementById('messages'); };
  function hideEmpty() { var e = document.getElementById('empty'); if (e) e.remove(); }

  function add(role, html, extra, asHtml) {
    hideEmpty();
    Tools.endGroup();
    var d = document.createElement('div');
    d.className = 'msg ' + role + (extra ? ' ' + extra : '');
    var labels = { user: 'You', agent: 'Agent' };
    d.innerHTML = '<div class="msg-label">'+(labels[role] || '')+'</div><div class="msg-body"></div>';
    pane().appendChild(d);
    var body = d.querySelector('.msg-body');
    if (html) {
      if (asHtml) body.innerHTML = html;
      else        body.textContent = html;
    }
    Scroll.maybe();
    return body;
  }

  function sys(text, variant) { add('sys', text, variant || ''); }
  return { add, sys, pane };
})();


// ═══════════════════════════════════════════════════════════════
// STREAMING AGENT MESSAGE
// ─────────────────────────────────────────────────────────────
// KEY FIX: flush() uses textContent (O(1)) so the main thread
// never blocks during streaming.  Full MD.render() is deferred
// to finalize(), wrapped in requestAnimationFrame() to yield
// control back to the browser before doing the parse.
// ═══════════════════════════════════════════════════════════════
const Stream = (() => {
  var el    = null;
  var buf   = '';
  var timer = null;

  function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  function _addCursor(body) {
    body.appendChild(Object.assign(document.createElement('span'), { className: 'cursor' }));
  }

  function start() {
    buf = '';
    el  = Messages.add('agent', '');
    el.classList.add('streaming');
    _addCursor(el);
  }

  function token(tok) {
    buf += tok;
    clearTimeout(timer);
    timer = setTimeout(flush, 60);
  }

  // Lightweight flush — plain text only, no markdown parsing
  function flush() {
    if (!el) return;
    el.textContent = buf;   // fast O(1), never blocks
    _addCursor(el);
    Scroll.maybe();
  }

  // Full markdown render, deferred via rAF so browser stays responsive
  function _renderFinal(body, rawBuf) {
    requestAnimationFrame(function() {
      var fixed = fixMojibake(rawBuf);
      if (!fixed.trim()) {
        var wrap = body.closest('.msg');
        if (wrap) wrap.remove();
      } else {
        body.classList.remove('streaming');
        body.innerHTML = MD.render(fixed) || _esc(fixed);
      }
      Scroll.maybe();
    });
  }

  function finalize() {
    clearTimeout(timer);
    if (!el) return;
    var body   = el;
    var rawBuf = buf;
    el  = null;
    buf = '';
    _renderFinal(body, rawBuf);
  }

  // reset() called on Stop — still render what we have rather than discard
  function reset() {
    clearTimeout(timer);
    if (!el) return;
    var body   = el;
    var rawBuf = buf;
    el  = null;
    buf = '';
    if (rawBuf.trim()) {
      _renderFinal(body, rawBuf);
    } else {
      var wrap = body.closest('.msg');
      if (wrap) wrap.remove();
    }
  }

  function active() { return el !== null; }
  return { start, token, finalize, reset, active };
})();


// ═══════════════════════════════════════════════════════════════
// TOOL ROWS
// ═══════════════════════════════════════════════════════════════
const Tools = (() => {
  var MAX = 12;
  var group     = null;
  var count     = 0;
  var pending   = new Map();
  var _esc = function(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); };

  function ensureGroup() {
    if (group) return group;
    var e = document.getElementById('empty'); if (e) e.remove();
    group = document.createElement('div');
    group.className = 'tool-group';
    Messages.pane().appendChild(group);
    count = 0;
    return group;
  }

  function endGroup() { group = null; count = 0; }

  function makeRow(name, args) {
    var row     = document.createElement('div');
    row.className = 'tool-row';
    row.dataset.s = 'pending';
    var preview = args ? (args.slice(0, 50) + (args.length > 50 ? '\u2026' : '')) : '';
    row.innerHTML = '<span class="tr-icon">\u25cc</span><span class="tr-name">'+_esc(name)+'</span>'+(preview ? '<span class="tr-args">('+_esc(preview)+')</span>' : '');
    if (!pending.has(name)) pending.set(name, []);
    pending.get(name).push(row);
    return row;
  }

  function call(name, args) {
    var g = ensureGroup();
    count++;
    if (count > MAX) {
      var more = g.querySelector('.tool-more');
      if (!more) {
        more = Object.assign(document.createElement('div'), { className: 'tool-more' });
        more.onclick = function() {
          more.remove();
          g.querySelectorAll('[data-hidden]').forEach(function(el){ delete el.dataset.hidden; el.style.display = ''; });
        };
        g.appendChild(more);
      }
      var n = count - MAX;
      more.textContent = '+'+n+' more call'+(n !== 1 ? 's' : '')+' \u2014 click to expand';
      var row = makeRow(name, args);
      row.dataset.hidden = '1'; row.style.display = 'none';
      g.insertBefore(row, more);
    } else {
      g.appendChild(makeRow(name, args));
    }
    Scroll.maybe();
  }

  function resolve(name, ok) {
    var q = pending.get(name);
    if (!q || !q.length) return;
    var row = q.shift();
    if (!q.length) pending.delete(name);
    if (!row) return;
    row.dataset.s = ok ? 'ok' : 'fail';
    row.querySelector('.tr-icon').textContent = ok ? '\u2713' : '\u2717';
  }

  function reset() { pending.clear(); endGroup(); }
  return { call, resolve, endGroup, reset };
})();


// ═══════════════════════════════════════════════════════════════
// AGENT STATE MACHINE
// ═══════════════════════════════════════════════════════════════
const Agent = (() => {
  var state     = 'idle';
  var sessionId = null;
  var requestId = null;
  var _hadContent = false;

  function setSession(id) {
    sessionId = id;
    var pill = document.getElementById('session-pill');
    if (id) { pill.textContent = id.slice(-8); pill.style.display = ''; }
    else    { pill.style.display = 'none'; }
  }

  function getSession() { return sessionId; }
  function running()    { return state !== 'idle'; }

  function lockUI(on) {
    document.getElementById('msg-input').disabled = on;
    var btn = document.getElementById('send-btn');
    btn.textContent = on ? 'Stop' : 'Send';
    btn.className   = on ? 'stop' : '';
    if (!on) { Scroll.pin(); Scroll.update(); }
  }

  function transition(next) {
    state = next;
    lockUI(next !== 'idle');
    Scroll.update();
  }

  function handleEvent(evt) {
    // If server says idle but we think we're running/waiting → clean up
    if (evt.type === 'connect') {
      var d = evt.data || {};
      if (d.session) setSession(d.session);
      if (d.mode)    Status.mode(d.mode);

      if (d.history && d.history.length) {
        var pane = document.getElementById('messages');
        if (!pane.querySelectorAll('.msg').length) {
          Replay.run(d.history);
        }
      }

      if (d.state === 'running' && state === 'idle') {
        transition('running');
        Status.set('running\u2026', 'run');
      } else if (d.state === 'waiting' && state === 'idle') {
        transition('waiting');
        Status.set('waiting \u2014 scheduled resume\u2026', 'wait');
        Messages.sys('\u23f0 Session waiting \u2014 will resume automatically', 'warn');
      } else if (d.state === 'idle' && state !== 'idle') {
        // SSE reconnect: server is idle but we're stuck → recover cleanly
        Stream.finalize();
        Tools.endGroup();
        transition('idle');
        Status.set('done', 'done');
        Status.iter('');
        Messages.sys('\u2713 task finished', 'ok');
        document.getElementById('msg-input').focus();
      }
      return;
    }

    if (state === 'waiting' && evt.type !== 'connect') {
      transition('running');
      Tools.reset();
      Stream.reset();
    }

    switch (evt.type) {

      case 'token':
        Tools.endGroup();
        if (!Stream.active()) Stream.start();
        Stream.token(evt.data);
        _hadContent = true;
        break;

      case 'session':
        setSession(evt.data);
        break;

      case 'tool': {
        var t = evt.data;
        if (t.startsWith('\u2713') || t.startsWith('\u2717')) {
          Tools.resolve(t.slice(1).trim(), t.startsWith('\u2713'));
        } else {
          Stream.finalize();
          var m = t.match(/^([^(]+)\(?([\s\S]*?)\)?$/);
          var name = m ? m[1].trim() : t;
          var args = m ? m[2].trim() : '';
          Tools.call(name, args);
          _hadContent = true;
        }
        break;
      }

      case 'status':
        Status.set(evt.data, 'run');
        break;

      case 'iteration': {
        var mi = String(evt.data).match(/(\d+)\/(\d+)/);
        if (mi) {
          Stream.finalize();
          Tools.endGroup();
          if (_hadContent) {
            Messages.sys('\u2014 iteration '+mi[1]+'/'+mi[2]+' \u2014', 'info');
          }
          _hadContent = false;
          Status.iter(mi[1]+'/'+mi[2]);
        }
        break;
      }

      case 'done':
        onDone(evt.data);
        break;

      case 'error':
        Stream.finalize();
        Tools.endGroup();
        Messages.sys('\u26a0 ' + evt.data, 'err');
        transition('idle');
        Status.set('error', 'err');
        Status.iter('');
        break;
    }
  }

  function onDone(reason) {
    Stream.finalize();
    Tools.endGroup();
    Status.iter('');

    var isWait = String(reason || '').startsWith('waiting');
    var isErr  = reason === 'error' || reason === 'stopped';

    if (isWait) {
      transition('waiting');
      Status.set('waiting \u2014 will resume automatically', 'wait');
      Messages.sys('\u23f8 ' + reason, 'warn');
    } else {
      transition('idle');
      Status.set(reason || 'done', isErr ? 'err' : 'done');

      if (isErr) {
        if (reason === 'stopped') {
          Messages.sys('\u2014 stopped \u2014', 'warn');
        }
      } else {
        // Normalise label — strip trailing " ✓" if already present to avoid dupe
        var raw   = reason || 'done';
        var label = raw.replace(/\s*\u2713\s*$/, '').trim() || 'task finished';
        Messages.sys('\u2713 ' + label, 'ok');
      }

      document.getElementById('msg-input').focus();
    }
  }

  async function send(text) {
    if (state !== 'idle') return;

    Messages.add('user', text);
    Scroll.pin();
    transition('running');
    Status.set('thinking\u2026', 'run');
    Tools.reset();
    _hadContent = false;

    requestId = 'r' + Date.now() + Math.random().toString(36).slice(2, 6);

    try {
      var resp = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, session_id: sessionId, request_id: requestId }),
      });
      if (!resp.ok) {
        var err = await resp.json().catch(function(){ return { error: 'HTTP ' + resp.status }; });
        throw new Error(err.error || 'HTTP ' + resp.status);
      }
    } catch (err) {
      Stream.finalize();
      Tools.endGroup();
      Messages.sys('Failed to send: ' + err.message, 'err');
      transition('idle');
      Status.set('error', 'err');
      Status.iter('');
    }
  }

  function stop() {
    if (state === 'waiting') {
      transition('idle');
      Status.set('wait dismissed', '');
      Messages.sys('\u2014 wait dismissed (session saved) \u2014', 'warn');
      return;
    }
    if (state !== 'running') return;

    if (requestId) {
      fetch('/stop', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ request_id: requestId }),
      }).catch(function(){});
      requestId = null;
    }
    Stream.reset();
    Tools.endGroup();
    Messages.sys('\u2014 stopped \u2014', 'warn');
    transition('idle');
    Status.set('stopped', '');
    Status.iter('');
  }

  async function newSession() {
    if (state !== 'idle') { stop(); return; }
    try { await fetch('/new', { method: 'POST' }); } catch (_) {}
    setSession(null);
    requestId   = null;
    _hadContent = false;
    Tools.reset();
    Stream.reset();
    document.getElementById('messages').innerHTML =
      '<div id="replay-banner"></div><div id="empty"><div class="empty-glyph">&#955;_</div><p>Send a message to start</p><div class="hint">/ for commands</div></div>';
    Status.set('ready', '');
    Status.iter('');
    document.getElementById('msg-input').focus();
  }

  function sendOrStop() {
    if (state !== 'idle') stop(); else _triggerSend();
  }

  function _triggerSend() {
    var inp  = document.getElementById('msg-input');
    var text = inp.value.trim();
    if (!text) return;
    inp.value = '';
    inp.style.height = 'auto';
    Palette.close();
    if (text.startsWith('/')) {
      Messages.add('user', text);
      SlashCmds.run(text);
    } else {
      send(text);
    }
  }

  return { sendOrStop, newSession, running, getSession, setSession, _handle: handleEvent };
})();


// ═══════════════════════════════════════════════════════════════
// PERSISTENT EVENTSOURCE
// ═══════════════════════════════════════════════════════════════
(function initStream() {
  var es      = null;
  var retryMs = 1000;

  function connect() {
    ConnDot.try();
    es = new EventSource('/stream');

    es.onopen = function() {
      ConnDot.ok();
      retryMs = 1000;
    };

    es.onmessage = function(e) {
      if (!e.data || e.data === '[DONE]') return;
      try {
        Agent._handle(JSON.parse(e.data));
      } catch (_) {}
    };

    es.onerror = function() {
      es.close();
      ConnDot.err();
      setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, 30000);
    };
  }

  connect();
})();


// ═══════════════════════════════════════════════════════════════
// SLASH COMMANDS
// ═══════════════════════════════════════════════════════════════
const CMDS = [
  { cmd: '/help',     desc: 'Show commands' },
  { cmd: '/new',      desc: 'Start a fresh session' },
  { cmd: '/sessions', desc: 'Browse history' },
  { cmd: '/tools',    desc: 'View all tools' },
  { cmd: '/status',   desc: 'Connection info' },
  { cmd: '/mode',     desc: 'Set permission mode', arg: '<auto|normal|manual>' },
  { cmd: '/soul',     desc: 'Agent personality' },
  { cmd: '/todo',     desc: 'Session todos' },
  { cmd: '/plan',     desc: 'Active plan' },
  { cmd: '/session',  desc: 'Current session ID' },
];

const HELP_TEXT = 'Slash commands:\n  /help               This text\n  /new                Fresh session\n  /sessions           Browse sessions\n  /tools              Tool list + MCP\n  /status             Config info\n  /mode auto|normal|manual\n  /soul               Agent personality\n  /todo               Current todos\n  /plan               Current plan\n  /session            Session ID';

const SlashCmds = (() => {
  async function run(text) {
    var parts = text.trim().split(/\s+/);
    var cmd   = parts[0].toLowerCase();
    var arg   = parts.slice(1).join(' ');
    var sys   = Messages.sys.bind(Messages);

    switch (cmd) {
      case '/help':     sys(HELP_TEXT, 'info'); break;
      case '/new':      Agent.newSession(); break;
      case '/sessions': UI.toggleSessions(); break;
      case '/tools':    UI.toggleTools(); break;
      case '/session':
        sys(Agent.getSession() ? 'Session: ' + Agent.getSession() : 'No active session.', 'info');
        break;
      case '/status':
        try {
          var s = await fetch('/status').then(function(r){ return r.json(); });
          sys('Workspace : '+s.workspace+'\nLLM       : '+s.llm_url+'\nSession   : '+(s.current_session ? s.current_session.slice(-8) : 'none')+'\nMode      : '+s.permission_mode+'\nMCP       : '+s.mcp_clients+' server'+(s.mcp_clients !== 1 ? 's' : ''), 'info');
        } catch (e) { sys('Failed: ' + e.message, 'err'); }
        break;
      case '/soul':
        try {
          var r = await fetch('/soul').then(function(r){ return r.json(); });
          sys(r.soul || '(no soul config)', 'info');
        } catch (e) { sys('Failed: ' + e.message, 'err'); }
        break;
      case '/mode': {
        if (!arg) { sys('Usage: /mode auto|normal|manual', 'warn'); break; }
        try {
          var r2 = await fetch('/mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: arg }),
          }).then(function(r){ return r.json(); });
          if (r2.ok) { Status.mode(arg); sys('Mode \u2192 ' + arg, 'ok'); }
          else       { sys('Invalid mode \''+arg+'\'. Use: auto, normal, manual', 'err'); }
        } catch (e) { sys('Failed: ' + e.message, 'err'); }
        break;
      }
      case '/todo':
      case '/plan': {
        var sid = Agent.getSession();
        if (!sid) { sys('No active session.', 'warn'); break; }
        try {
          var r3 = await fetch('/cmd', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmd: cmd.slice(1), session_id: sid }),
          }).then(function(r){ return r.json(); });
          sys(r3.text || '(empty)', 'info');
        } catch (e) { sys('Failed: ' + e.message, 'err'); }
        break;
      }
      default:
        sys('Unknown command: ' + cmd + '. Type /help for list.', 'err');
    }
  }
  return { run };
})();


// ═══════════════════════════════════════════════════════════════
// COMMAND PALETTE
// ═══════════════════════════════════════════════════════════════
const Palette = (() => {
  var idx = -1;
  var pal = function(){ return document.getElementById('palette'); };
  var _esc = function(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); };

  function build(val) {
    var q = val.slice(1).toLowerCase();
    var matches = q === '' ? CMDS : CMDS.filter(function(c){ return c.cmd.includes(q) || c.desc.toLowerCase().includes(q); });
    if (!matches.length) { close(); return; }
    pal().innerHTML = '';
    matches.forEach(function(c, i) {
      var d = document.createElement('div');
      d.className = 'pi' + (i === idx ? ' sel' : '');
      d.innerHTML = '<span class="pi-cmd">'+_esc(c.cmd)+(c.arg ? ' <span style="color:var(--text3)">'+_esc(c.arg)+'</span>' : '')+'</span><span class="pi-desc">'+_esc(c.desc)+'</span>';
      d.onclick = function(){ select(c.cmd); };
      pal().appendChild(d);
    });
    pal().classList.add('open');
  }

  function close() { pal().classList.remove('open'); idx = -1; }

  function select(cmd) {
    var inp = document.getElementById('msg-input');
    inp.value = cmd === '/mode' ? '/mode ' : cmd;
    close();
    inp.focus();
  }

  function onInput(el) {
    autoResize(el);
    if (el.value.startsWith('/')) build(el.value);
    else close();
  }

  function onKey(e) {
    var p = pal();
    if (p.classList.contains('open')) {
      var items = p.querySelectorAll('.pi');
      if (e.key === 'ArrowDown') { e.preventDefault(); idx = Math.min(idx+1, items.length-1); items.forEach(function(el,i){ el.classList.toggle('sel', i===idx); }); return; }
      if (e.key === 'ArrowUp')   { e.preventDefault(); idx = Math.max(idx-1, 0);              items.forEach(function(el,i){ el.classList.toggle('sel', i===idx); }); return; }
      if (e.key === 'Tab' || (e.key === 'Enter' && idx >= 0)) { e.preventDefault(); (idx >= 0 ? items[idx] : items[0]) && (idx >= 0 ? items[idx] : items[0]).click(); return; }
      if (e.key === 'Escape') { close(); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); Agent.sendOrStop(); }
  }

  return { onInput, onKey, close };
})();

function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 140) + 'px'; }


// ═══════════════════════════════════════════════════════════════
// SESSIONS PANEL
// ═══════════════════════════════════════════════════════════════
const SessionsPanel = (() => {
  var _esc = function(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); };

  async function load() {
    var list = document.getElementById('sessions-list');
    list.innerHTML = '<div style="padding:10px;color:var(--text3);font-size:11px">Loading\u2026</div>';
    try {
      var sessions = await fetch('/sessions').then(function(r){ return r.json(); });
      list.innerHTML = '';
      if (!sessions.length) { list.innerHTML = '<div style="padding:10px;color:var(--text3);font-size:11px">No sessions yet</div>'; return; }
      sessions.forEach(function(s) {
        var d = document.createElement('div');
        d.className = 'session-item' + (s.id === Agent.getSession() ? ' active' : '');
        d.innerHTML = '<div class="si-id">'+_esc(s.id)+'</div><div class="si-task">'+_esc(s.task||'\u2014')+'</div><div class="si-stat '+_esc(s.status)+'">'+_esc(s.status)+' \u00b7 '+s.iterations+' iter</div>';
        d.onclick = function() {
          if (Agent.running()) return;
          Agent.setSession(s.id);
          UI.closeSessions();
          Messages.sys('resumed ' + s.id.slice(-8));
        };
        list.appendChild(d);
      });
    } catch (err) {
      list.innerHTML = '<div style="padding:10px;color:var(--red);font-size:11px">Error: '+_esc(err.message)+'</div>';
    }
  }
  return { load };
})();


// ═══════════════════════════════════════════════════════════════
// TOOLS PANEL
// ═══════════════════════════════════════════════════════════════
const ToolsPanel = (() => {
  var data = null;
  var _esc = function(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); };

  async function load() {
    var list = document.getElementById('tools-list');
    list.innerHTML = '<div style="padding:10px;color:var(--text3);font-size:11px">Loading\u2026</div>';
    try {
      data = await fetch('/tools').then(function(r){ return r.json(); });
      render('');
    } catch (err) {
      list.innerHTML = '<div style="padding:10px;color:var(--red);font-size:11px">Error: '+_esc(err.message)+'</div>';
    }
  }

  function render(q) {
    var list = document.getElementById('tools-list');
    list.innerHTML = '';
    var qlo  = q.toLowerCase();
    var all  = Object.assign({}, data.builtin, data.mcp || {});
    var shown = false;
    Object.entries(all).forEach(function(entry) {
      var cat   = entry[0];
      var tools = entry[1];
      var filtered = tools.filter(function(t){ return !qlo || t.name.includes(qlo) || t.description.toLowerCase().includes(qlo); });
      if (!filtered.length) return;
      shown = true;
      var isMcp  = cat.startsWith('MCP');
      var catEl  = document.createElement('div');
      catEl.className = 'tool-cat' + (isMcp ? ' mcp-cat' : '');
      catEl.innerHTML  = '<div class="tool-cat-hdr">'+_esc(cat)+'</div>';
      filtered.forEach(function(t) {
        var params = t.params.length ? t.params.map(function(p){ return t.required.includes(p) ? p : '['+p+']'; }).join(', ') : '';
        var entry2  = document.createElement('div');
        entry2.className = 'tool-entry';
        entry2.innerHTML  = '<div class="te-name">'+_esc(t.name)+'<span class="te-params">'+(params ? '('+_esc(params)+')' : '')+'</span></div>'+(t.description ? '<div class="te-desc">'+_esc(t.description)+'</div>' : '');
        catEl.appendChild(entry2);
      });
      list.appendChild(catEl);
    });
    if (!shown) list.innerHTML = '<div style="padding:10px;color:var(--text3);font-size:11px">No matching tools</div>';
  }

  function filter(q) { if (data) render(q); }
  return { load, filter };
})();


// ═══════════════════════════════════════════════════════════════
// UI PANEL MANAGEMENT
// ═══════════════════════════════════════════════════════════════
const UI = (() => {
  var overlay = function(){ return document.getElementById('overlay'); };
  function openPanel(id)  { document.getElementById(id).classList.add('open');    overlay().classList.add('show'); }
  function closePanel(id) { document.getElementById(id).classList.remove('open'); overlay().classList.remove('show'); }
  function isOpen(id)     { return document.getElementById(id).classList.contains('open'); }

  async function toggleSessions() {
    if (isOpen('sessions-panel')) { closeSessions(); return; }
    closeTools();
    await SessionsPanel.load();
    openPanel('sessions-panel');
  }
  function closeSessions() { closePanel('sessions-panel'); }

  async function toggleTools() {
    if (isOpen('tools-panel')) { closeTools(); return; }
    closeSessions();
    await ToolsPanel.load();
    openPanel('tools-panel');
  }
  function closeTools() { closePanel('tools-panel'); }
  function closeAll()   { closeSessions(); closeTools(); }
  return { toggleSessions, closeSessions, toggleTools, closeTools, closeAll };
})();


// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
(async function() {
  try {
    var s = await fetch('/status').then(function(r){ return r.json(); });
    Status.mode(s.permission_mode);
    if (s.current_session) Agent.setSession(s.current_session);
  } catch (_) {}
})();
</script>
</body>
</html>"""


# =============================================================================
# SCHEDULER
# =============================================================================

def _scheduler_loop():
    global _current_session_id

    state_mgr   = StateManager(WORKSPACE)
    session_mgr = SessionManager(WORKSPACE)
    _running: set = set()
    _running_lock = threading.Lock()

    def _run_wake(sid: str):
        global _current_session_id
        with _running_lock:
            _running.add(sid)

        last_status = [""]

        def _do_run():
            global _current_session_id
            _set_agent_state("running")
            _filt = _ThinkingFilter()

            def _token_cb(tok: str) -> None:
                _filt.feed(tok)

            _tl.token_cb = _token_cb

            def _event_cb(event):
                _push_event(event, None, last_status, filt=_filt)

            try:
                saved = state_mgr.load(sid)
                ws = WaitState.from_dict(saved.wait_state)
                wake_task = (
                    f"Your scheduled wait has now completed. "
                    f"Original reason for waiting: \"{ws.reason}\". "
                    f"Do NOT output another WAIT — the timer has expired. "
                    f"Proceed directly with the next step of the task."
                )

                result = run_agent(
                    task            = wake_task,
                    workspace       = WORKSPACE,
                    permission_mode = PermissionMode.AUTO,
                    resume_session  = sid,
                    mode            = "interactive",
                    event_callback  = _event_cb,
                    soul            = soul,
                )
                old_key = _current_session_id
                with _session_lock:
                    _current_session_id = result.session_id
                _chatlog_merge_keys(old_key, result.session_id)

                _broadcast(("session", result.session_id))

                if result.status == "waiting":
                    _set_agent_state("waiting")
                else:
                    _set_agent_state("idle")

                label = {
                    "completed":      "done \u2713",
                    "waiting":        f"waiting until {result.wait_until}",
                    "max_iterations": "max iterations reached",
                    "error":          "error",
                    "interrupted":    "interrupted",
                }.get(result.status, result.status)
                _filt.flush()
                _broadcast(("done", label))

            except Exception as exc:
                _broadcast(("error", str(exc)[:200]))
                _broadcast(("done", "error"))
                _set_agent_state("idle")
            finally:
                _tl.token_cb = None
                with _running_lock:
                    _running.discard(sid)

        acquired = _AGENT_LOCK.acquire(timeout=_AGENT_LOCK_TIMEOUT)
        if not acquired:
            _broadcast(("error", "agent lock timeout \u2014 server may need restart"))
            _broadcast(("done", "error"))
            with _running_lock:
                _running.discard(sid)
            return
        try:
            _do_run()
        finally:
            _AGENT_LOCK.release()

    while True:
        try:
            for session in session_mgr.list_recent(200):
                if session.get("status") != "waiting":
                    continue
                sid = session["id"]
                with _running_lock:
                    if sid in _running:
                        continue
                saved = state_mgr.load(sid)
                if not saved or not saved.wait_state:
                    try:
                        data = session_mgr.load(sid)
                        if data:
                            msgs, meta = data
                            meta["status"] = "idle"
                            session_mgr.save(sid, msgs, meta)
                    except Exception:
                        pass
                    continue
                ws = WaitState.from_dict(saved.wait_state)
                if not ws.is_ready():
                    continue
                threading.Thread(
                    target=_run_wake, args=(sid,),
                    daemon=True, name=f"wake-{sid[-8:]}",
                ).start()
        except Exception:
            pass
        time.sleep(10)


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream")
def stream():
    q = _register_stream_q()

    def generate():
        with _agent_state_lock:
            ag_state = _agent_state
        with _session_lock:
            sid = _current_session_id

        history = _chatlog_get(sid)

        connect_payload = {
            "state":   ag_state,
            "session": sid,
            "mode":    _current_permission_mode,
            "history": history,
        }
        yield f"data: {json.dumps({'type': 'connect', 'data': connect_payload}, ensure_ascii=False)}\n\n"

        KA   = 15
        last = time.time()
        try:
            while True:
                timeout = max(0.3, KA - (time.time() - last))
                try:
                    item = q.get(timeout=timeout)
                except queue.Empty:
                    yield ": ka\n\n"
                    last = time.time()
                    continue
                kind, payload = item
                yield f"data: {json.dumps({'type': kind, 'data': payload}, ensure_ascii=False)}\n\n"
                last = time.time()
        except GeneratorExit:
            pass
        finally:
            _unregister_stream_q(q)

    return _sse_response(generate())


@app.route("/chat", methods=["POST"])
def chat():
    global _current_session_id

    data     = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    req_sid  = data.get("session_id") or None
    rid      = data.get("request_id") or f"req_{time.time()}"

    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    with _session_lock:
        session_id = req_sid or _current_session_id

    _chatlog_append(("user_msg", user_msg))

    stop_event = threading.Event()
    with _stop_events_lock:
        _stop_events[rid] = stop_event

    last_status = [""]

    def run_in_thread():
        global _current_session_id
        _set_agent_state("running")
        _filt = _ThinkingFilter()

        def _token_cb(tok: str) -> None:
            if stop_event.is_set():
                raise _AgentStopped("stopped by user")
            _filt.feed(tok)

        _tl.token_cb = _token_cb

        def event_cb(event):
            _push_event(event, stop_event, last_status, filt=_filt)

        try:
            result = run_agent(
                task            = user_msg,
                workspace       = WORKSPACE,
                permission_mode = PermissionMode.AUTO,
                resume_session  = session_id,
                mode            = "interactive",
                event_callback  = event_cb,
                soul            = soul,
            )

            if not stop_event.is_set():
                old_key = _current_session_id
                with _session_lock:
                    _current_session_id = result.session_id
                _chatlog_merge_keys(old_key, result.session_id)

                _broadcast(("session", result.session_id))

                if result.status == "waiting":
                    _set_agent_state("waiting")
                else:
                    _set_agent_state("idle")

                label = {
                    "completed":      "done \u2713",
                    "waiting":        f"waiting until {result.wait_until}",
                    "max_iterations": "max iterations reached",
                    "error":          "error",
                    "interrupted":    "interrupted",
                    "cancelled":      "cancelled",
                }.get(result.status, result.status)
                _filt.flush()
                _broadcast(("done", label))
            else:
                old_key = _current_session_id
                with _session_lock:
                    _current_session_id = result.session_id
                _chatlog_merge_keys(old_key, result.session_id)
                _broadcast(("session", result.session_id))
                _set_agent_state("idle")
                _broadcast(("done", "stopped"))

        except _AgentStopped:
            _set_agent_state("idle")
            _broadcast(("done", "stopped"))

        except Exception as exc:
            if not stop_event.is_set():
                _broadcast(("error", str(exc)[:200]))
            _set_agent_state("idle")
        finally:
            _tl.token_cb = None
            with _stop_events_lock:
                _stop_events.pop(rid, None)

    def locked_run():
        acquired = _AGENT_LOCK.acquire(timeout=_AGENT_LOCK_TIMEOUT)
        if not acquired:
            _broadcast(("error", "agent busy \u2014 try again in a moment"))
            _broadcast(("done", "error"))
            _set_agent_state("idle")
            with _stop_events_lock:
                _stop_events.pop(rid, None)
            return
        try:
            run_in_thread()
        finally:
            _AGENT_LOCK.release()

        if _get_agent_state() == "running":
            _set_agent_state("idle")
            _broadcast(("done", "error"))

    threading.Thread(target=locked_run, daemon=True, name="agent-chat").start()
    return jsonify({"ok": True, "request_id": rid})


@app.route("/sessions")
def list_sessions():
    return jsonify(SessionManager(WORKSPACE).list_recent(20))


@app.route("/status")
def status():
    with _session_lock:
        sid = _current_session_id
    return jsonify({
        "workspace":       str(WORKSPACE),
        "llm_url":         Config.LLM_URL,
        "current_session": sid,
        "permission_mode": _current_permission_mode,
        "mcp_clients":     len(_global_mcp.clients),
    })


@app.route("/soul")
def get_soul():
    return jsonify({"soul": soul or "(no soul config \u2014 create .soul.md in workspace)"})


@app.route("/mode", methods=["POST"])
def set_mode():
    global _current_permission_mode
    data = request.get_json(force=True, silent=True) or {}
    mode = (data.get("mode") or "").strip().lower()
    if mode not in {"auto", "normal", "manual"}:
        return jsonify({"ok": False, "error": "Invalid mode. Choose: auto, normal, manual"})
    _current_permission_mode = mode
    return jsonify({"ok": True, "mode": mode})


@app.route("/tools")
def list_tools():
    return jsonify(_build_tools_payload())


@app.route("/cmd", methods=["POST"])
def run_cmd():
    data       = request.get_json(force=True, silent=True) or {}
    cmd        = (data.get("cmd") or "").strip()
    session_id = data.get("session_id") or ""
    if not session_id:
        return jsonify({"text": "No active session."})

    if cmd == "todo":
        try:
            mgr    = TodoManager(WORKSPACE, session_id)
            result = mgr.list_all()
            todos  = result.get("todos", [])
            if not todos:
                return jsonify({"text": "No todos yet."})
            icons = {"pending": "\u23f3", "in_progress": "\U0001f504", "completed": "\u2705", "blocked": "\U0001f6ab"}
            lines = [f"Todos ({result['completed']}/{result['total']} done):"]
            for t in todos:
                lines.append(f"  {icons.get(t['status'],'?')} #{t['id']} [{t['status']}] {t['description']}")
                if t.get("notes"):
                    lines.append(f"      {t['notes']}")
            return jsonify({"text": "\n".join(lines)})
        except Exception as e:
            return jsonify({"text": f"Error: {e}"})

    if cmd == "plan":
        try:
            mgr = PlanManager(WORKSPACE, session_id)
            ctx = mgr.get_context()
            return jsonify({"text": ctx or "No active plan."})
        except Exception as e:
            return jsonify({"text": f"Error: {e}"})

    return jsonify({"text": f"Unknown command: {cmd}"})


@app.route("/new", methods=["POST"])
def new_session():
    global _current_session_id
    with _session_lock:
        old = _current_session_id
        _current_session_id = None
    _chatlog_clear(old)
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop_agent():
    data = request.get_json(force=True, silent=True) or {}
    rid  = data.get("request_id") or ""
    with _stop_events_lock:
        ev = _stop_events.get(rid)
    if ev:
        ev.set()
        return jsonify({"stopped": True})
    return jsonify({"stopped": False, "detail": "unknown request_id"})


# =============================================================================
# STARTUP
# =============================================================================

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


if __name__ == "__main__":
    port      = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    ip        = _local_ip()
    mcp_count = len(_global_mcp.clients)

    threading.Thread(target=_scheduler_loop, daemon=True, name="web-scheduler").start()

    print("\n" + "\u2550" * 58)
    print("  LMAgent Web  v6.5.2")
    print("\u2550" * 58)
    print(f"  Local  \u2192  http://localhost:{port}")
    print(f"  Phone  \u2192  http://{ip}:{port}")
    print(f"  Workspace : {WORKSPACE}")
    print(f"  LLM       : {Config.LLM_URL}")
    print(f"  MCP       : {mcp_count} server{'s' if mcp_count != 1 else ''} loaded")
    print("\u2550" * 58)
    print("  FIX: Non-blocking stream rendering (no more page freeze) \u2713")
    print("\u2550" * 58 + "\n")

    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
