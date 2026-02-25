"""
LMAgent Web — v6.6.1
------------------------------------
  agent_core.py   — Config, logging, session/state/todo/plan managers, etc.
  agent_tools.py  — Tool handlers, LLMClient, _HeaderStreamCb, TOOL_SCHEMAS
  agent_main.py   — run_agent()

Place this file next to those three files.

Changes vs v6.6.0  (UX: friendly auth for desktop + mobile):
  - UX:  AGENT_TOKEN is now a short 6-digit PIN by default (easy to type
         on mobile). Set AGENT_TOKEN env var to override with any value.
  - UX:  Unauthorized page shows a QR code (generated client-side via
         qrcode.js from cdnjs — no new Python deps) plus a PIN entry form.
         Mobile: scan the QR from the computer screen.
         Desktop: type or click the PIN.
         qrcode.js added to CSP script-src allowlist.
  - Version badge updated to v6.6.1.
"""

import fnmatch
import functools
import json
import os
import queue
import re
import secrets
import socket
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request

# ── Import the agent ──────────────────────────────────────────────────────────
try:
    import agent_core
    import agent_tools
    import agent_main as _agent_main_mod

    from agent_core import (
        Config,
        Colors,
        colored,
        strip_thinking,
        MCPManager,
        PermissionMode,
        PlanManager,
        Safety,
        SessionManager,
        StateManager,
        WaitState,
        SoulConfig,
        TodoManager,
        Log,
    )
    from agent_tools import (
        TOOL_SCHEMAS,
        LLMClient,
        _REQUIRED_ARG_TOOLS,
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


@app.after_request
def _security_headers(response: "Response") -> "Response":
    """Attach security headers to every response."""
    h = response.headers
    h["X-Content-Type-Options"] = "nosniff"
    h["X-Frame-Options"]        = "DENY"
    h["Referrer-Policy"]        = "no-referrer"
    # unsafe-inline is required because the UI uses inline <script>/<style>.
    # Still blocks injected scripts from loading from external origins.
    h["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response


# =============================================================================
# SECURITY CONFIG
# =============================================================================

# Token: a short 6-digit PIN by default — easy to type on mobile.
# Set AGENT_TOKEN env var to use a custom value (string, any length).
_AGENT_TOKEN: str = os.environ.get("AGENT_TOKEN") or str(secrets.randbelow(900000) + 100000)

# Host: localhost by default. Set AGENT_HOST=0.0.0.0 to expose to LAN.
_HOST: str = os.environ.get("AGENT_HOST", "0.0.0.0")


# HTTPS: set both env vars to PEM file paths to enable TLS.
_SSL_CERT: str = os.environ.get("AGENT_CERT", "")
_SSL_KEY:  str = os.environ.get("AGENT_KEY",  "")

# ── Rate limiter (pure in-memory, no deps) ────────────────────────────────────
_rate_data: "dict[str, list]" = {}
_rate_lock = threading.Lock()
_RATE_LIMIT  = 120    # max requests per window per IP
_RATE_WINDOW = 60.0   # seconds


def _is_rate_limited(ip: str) -> bool:
    """Sliding-window rate limiter. Returns True when the IP is over limit."""
    now = time.time()
    with _rate_lock:
        ts = [t for t in _rate_data.get(ip, []) if now - t < _RATE_WINDOW]
        if len(ts) >= _RATE_LIMIT:
            _rate_data[ip] = ts
            return True
        ts.append(now)
        _rate_data[ip] = ts
        # Prune stale IPs to prevent unbounded memory growth.
        if len(_rate_data) > 1000:
            cutoff = now - _RATE_WINDOW * 2
            stale = [k for k, v in _rate_data.items() if not v or v[-1] < cutoff]
            for k in stale:
                del _rate_data[k]
        return False


def _require_auth(f):
    """Decorator: enforce token auth + rate limiting on a route.

    Accepts the token from:
      - X-Token request header  (fetch / XHR)
      - ?token= query param     (EventSource, direct URL)
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if _AGENT_TOKEN:
            token = (request.headers.get("X-Token", "")
                     or request.args.get("token", ""))
            # secrets.compare_digest prevents timing-based token oracle attacks.
            if not token or not secrets.compare_digest(token, _AGENT_TOKEN):
                return jsonify({"error": "unauthorized"}), 401
        ip = request.remote_addr or "unknown"
        if _is_rate_limited(ip):
            return jsonify({"error": "rate limited — try again shortly"}), 429
        return f(*args, **kwargs)
    return wrapper


# Shown to visitors who hit / without a valid token.
# __AUTH_URL__ is replaced server-side with the full URL including token.
_UNAUTH_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LMAgent — Sign In</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0d0f;color:#8e8c88;font-family:'IBM Plex Mono',monospace;
       display:flex;align-items:center;justify-content:center;
       min-height:100vh;padding:20px}
  .box{text-align:center;padding:36px 32px;max-width:360px;width:100%;
       background:#16161a;border:1px solid #252530;border-radius:12px}
  .mark{width:40px;height:40px;background:#e8a245;border-radius:8px;
        display:flex;align-items:center;justify-content:center;
        font-size:20px;color:#0d0d0f;font-weight:700;margin:0 auto 16px}
  h2{color:#ddd8d0;margin-bottom:6px;font-size:15px;font-weight:600}
  .sub{font-size:11px;color:#4e4c50;margin-bottom:22px}
  /* QR code */
  #qr{display:flex;justify-content:center;margin-bottom:20px}
  #qr canvas,#qr img{border-radius:8px;border:6px solid #fff}
  /* Divider */
  .divider{display:flex;align-items:center;gap:10px;margin:18px 0;font-size:10px;color:#4e4c50}
  .divider::before,.divider::after{content:'';flex:1;height:1px;background:#252530}
  /* PIN entry */
  .pin-row{display:flex;gap:8px;justify-content:center;margin-bottom:14px}
  .pin-digit{
    width:44px;height:54px;border-radius:8px;
    border:1px solid #252530;background:#0d0d0f;
    color:#ddd8d0;font-family:'IBM Plex Mono',monospace;
    font-size:22px;font-weight:600;text-align:center;
    outline:none;caret-color:#e8a245;
    transition:border-color .12s;
    -webkit-appearance:none;
  }
  .pin-digit:focus{border-color:#e8a245}
  .pin-digit.filled{border-color:#32323e;color:#e8a245}
  .go-btn{
    width:100%;padding:11px;border-radius:8px;
    background:#e8a245;border:none;color:#0d0d0f;
    font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;
    cursor:pointer;transition:background .12s;margin-bottom:10px;
  }
  .go-btn:hover{background:#f0b055}
  .go-btn:disabled{background:#252530;color:#4e4c50;cursor:default}
  .err{font-size:11px;color:#e05555;min-height:16px;margin-top:2px}
  .hint{font-size:10px;color:#4e4c50;margin-top:16px;line-height:1.6}
  code{background:#0d0d0f;border:1px solid #252530;padding:1px 6px;
       border-radius:3px;color:#3ecfb2;font-size:10px}
</style>
</head>
<body>
<div class="box">
  <div class="mark">&#955;</div>
  <h2>LMAgent</h2>
  <p class="sub">Scan with your phone or enter the PIN</p>

  <div id="qr"></div>

  <div class="divider">or enter PIN</div>

  <div class="pin-row" id="pin-row">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
  </div>
  <button class="go-btn" id="go-btn" disabled onclick="submitPin()">Unlock</button>
  <div class="err" id="err"></div>

  <div class="hint">
    PIN printed to server console on startup.<br>
    Set <code>AGENT_TOKEN</code> env var for a fixed PIN.
  </div>
</div>

<script>
(function() {
  // QR code — URL is injected server-side
  var authUrl = '__AUTH_URL__';
  try {
    new QRCode(document.getElementById('qr'), {
      text: authUrl,
      width: 200, height: 200,
      colorDark: '#000000', colorLight: '#ffffff',
      correctLevel: QRCode.CorrectLevel.M
    });
  } catch(e) {
    document.getElementById('qr').style.display = 'none';
  }

  // 6-digit PIN inputs — auto-advance, auto-submit, paste support
  var digits = Array.from(document.querySelectorAll('.pin-digit'));
  var btn = document.getElementById('go-btn');
  var err = document.getElementById('err');

  function pinValue() { return digits.map(function(d){return d.value}).join(''); }
  function updateBtn() {
    var full = pinValue().length === 6;
    btn.disabled = !full;
    digits.forEach(function(d){ d.classList.toggle('filled', d.value !== ''); });
  }

  digits.forEach(function(d, i) {
    d.addEventListener('input', function() {
      // Allow only digits
      d.value = d.value.replace(/\\D/g, '').slice(0, 1);
      updateBtn();
      if (d.value && i < digits.length - 1) digits[i+1].focus();
      if (pinValue().length === 6) submitPin();
    });
    d.addEventListener('keydown', function(e) {
      if (e.key === 'Backspace' && !d.value && i > 0) {
        digits[i-1].value = '';
        digits[i-1].focus();
        updateBtn();
      }
    });
    d.addEventListener('paste', function(e) {
      e.preventDefault();
      var text = (e.clipboardData || window.clipboardData).getData('text').replace(/\\D/g,'').slice(0,6);
      text.split('').forEach(function(ch, j){ if (digits[i+j]) digits[i+j].value = ch; });
      updateBtn();
      var next = Math.min(i + text.length, digits.length - 1);
      digits[next].focus();
      if (pinValue().length === 6) submitPin();
    });
  });

  digits[0].focus();

  window.submitPin = function() {
    var pin = pinValue();
    if (pin.length !== 6) return;
    btn.disabled = true;
    err.textContent = '';
    // Build redirect URL preserving host/path, adding token
    var base = window.location.origin + window.location.pathname;
    window.location.href = base + '?token=' + encodeURIComponent(pin);
  };
})();
</script>
</body>
</html>"""

# Reduced from 300 s — a stuck agent should surface quickly, not block for 5 min.
_AGENT_LOCK         = threading.Lock()
_AGENT_LOCK_TIMEOUT = 60

_tl = threading.local()

_session_lock            = threading.Lock()
_current_session_id: "str | None" = None
_current_permission_mode = Config.PERMISSION_MODE

_stop_events:      "dict[str, threading.Event]" = {}
_stop_events_lock = threading.Lock()

_whisper_store: "list[str]" = []
_whisper_lock  = threading.Lock()

_agent_state      = "idle"
_agent_state_lock = threading.Lock()


def _set_agent_state(s: str) -> None:
    global _agent_state
    with _agent_state_lock:
        _agent_state = s


def _get_agent_state() -> str:
    with _agent_state_lock:
        return _agent_state


# ── Permission mode helper ────────────────────────────────────────────────────

def _resolve_permission_mode() -> PermissionMode:
    """Map the current string mode to the PermissionMode enum, with AUTO fallback."""
    try:
        return PermissionMode[_current_permission_mode.upper()]
    except (KeyError, AttributeError):
        return PermissionMode.AUTO


# =============================================================================
# MONKEY-PATCH LLMClient._parse_stream
# =============================================================================

def _web_parse_stream(resp, stream_callback):
    content       = ""
    tool_calls: "dict[int, dict]" = {}
    next_idx      = 0
    finish_reason = None
    in_think      = False

    resp.encoding = "utf-8"
    _tl.current_resp = resp

    thinking_cb = getattr(_tl, "thinking_cb", None)
    stop_event  = getattr(_tl, "stop_event",  None)

    for line in resp.iter_lines(decode_unicode=True):
        if stop_event and stop_event.is_set():
            try:
                resp.close()
            except Exception:
                pass
            break

        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta  = choice.get("delta", {})

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        raw_content = delta.get("content")
        if raw_content:
            content += raw_content
            if stream_callback:
                for part in re.split(r"(</?think>)", raw_content, flags=re.IGNORECASE):
                    if not part:
                        continue
                    if part.lower() == "<think>":
                        in_think = True
                    elif part.lower() == "</think>":
                        in_think = False
                        thinking_cb = getattr(_tl, "thinking_cb", None)
                        if thinking_cb:
                            thinking_cb(None)
                    elif in_think:
                        sys.stdout.write(colored(part, Colors.GRAY))
                        sys.stdout.flush()
                        thinking_cb = getattr(_tl, "thinking_cb", None)
                        if thinking_cb:
                            thinking_cb(part)
                    else:
                        stream_callback(part)

        thinking_blocks = delta.get("thinking")
        if isinstance(thinking_blocks, list):
            for tb in thinking_blocks:
                tb_text = tb.get("thinking") if isinstance(tb, dict) else None
                if tb_text:
                    sys.stdout.write(colored(tb_text, Colors.GRAY))
                    sys.stdout.flush()
                    thinking_cb = getattr(_tl, "thinking_cb", None)
                    if thinking_cb:
                        thinking_cb(tb_text)
                        thinking_cb(None)

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
        Log.warning("\u26a0\ufe0f  Generation stopped: output token limit hit (finish_reason=length)")
        incomplete = True
    elif finish_reason == "stop" and incomplete:
        Log.warning("\u26a0\ufe0f  finish_reason=stop but tool calls were incomplete")
    elif finish_reason is None and incomplete:
        Log.warning("\u26a0\ufe0f  finish_reason=None \u2014 stream may have ended prematurely")

    clean_content, _ = strip_thinking(content)
    return {
        "content":       clean_content,
        "tool_calls":    calls or None,
        "incomplete":    incomplete,
        "finish_reason": finish_reason,
    }


LLMClient._parse_stream = staticmethod(_web_parse_stream)


# =============================================================================
# MOJIBAKE FIX
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
    """Broadcast an event to all registered SSE queues.

    Mojibake correction is applied to all string payloads.
    Queue list is snapshotted under lock then iterated outside it,
    minimising lock hold time on the hot token path.
    """
    kind, payload = item
    if isinstance(payload, str):
        payload = _fix_mojibake(payload)
    item = (kind, payload)
    _chatlog_append(item)
    # Snapshot — avoids holding the lock while doing put_nowait on every queue.
    with _stream_queues_lock:
        queues = list(_stream_queues)
    for q in queues:
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

_REPLAY_KINDS = {"token", "thinking", "tool", "status", "iteration", "done", "error", "session"}


def _chatlog_session_key() -> str:
    with _session_lock:
        return _current_session_id or "_none"


def _chatlog_append(item: tuple) -> None:
    kind, _ = item
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
    """Merge chat log from old_key into new_key atomically under a single lock."""
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
# THINKING CALLBACK HELPERS
# =============================================================================

def _make_thinking_helpers():
    buf = [""]

    def flush_thinking():
        text = buf[0].strip()
        if text:
            _broadcast(("thinking", text))
        buf[0] = ""

    def thinking_cb(part):
        if part is None:
            flush_thinking()
        else:
            buf[0] += part

    return thinking_cb, flush_thinking


# =============================================================================
# EVENT NOISE FILTER
# =============================================================================

_STATUS_NOISE = frozenset({
    "thinking model mode", "state restored", "soul loaded", "mcp ",
    "checking llm", "llm connected", "task:", "workspace :",
    "llm       :", "loaded ", "compacting context", "compacted:",
    "plan approved", "plan mode",
})


def _is_noisy(msg: str) -> bool:
    ml = msg.lower()
    return any(ml.startswith(kw) or kw in ml[:40] for kw in _STATUS_NOISE)


# =============================================================================
# SHARED EVENT HANDLER
# =============================================================================

def _push_event(event, stop_event: "threading.Event | None", last_status: list,
                flush_thinking=None) -> None:
    if stop_event and stop_event.is_set():
        return
    etype = event.type
    edata = event.data
    if etype == "tool_call":
        if flush_thinking:
            flush_thinking()
        name    = edata.get("name", "?")
        preview = edata.get("args_preview", "")[:50]
        _broadcast(("tool", f"{name}({preview})"))
    elif etype == "tool_result":
        mark = "\u2713" if edata.get("success") else "\u2717"
        _broadcast(("tool", f"{mark} {edata.get('name', '')}"))
    elif etype == "iteration":
        if flush_thinking:
            flush_thinking()
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
        _broadcast(("status", f"done \u2014 {edata.get('reason', '')}"))


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
        tools = [
            {
                "name":        t.get("name", ""),
                "description": t.get("description", ""),
                "params":      list(t.get("inputSchema", {}).get("properties", {}).keys()),
                "required":    t.get("inputSchema", {}).get("required", []),
            }
            for t in client.tools
        ]
        if tools:
            mcp_groups[f"MCP \u00b7 {client.name}"] = tools

    return {"builtin": builtin, "mcp": mcp_groups}


# =============================================================================
# HTML TEMPLATE  (v6.5.9)
# =============================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>LMAgent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
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

.msg.agent { position: relative; }
.msg.agent .msg-body {
  background: var(--agent-bg); border-color: var(--agent-bdr);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.75;
  color: var(--text);
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

.msg.agent .msg-body code {
  background: var(--code-bg); border: 1px solid var(--code-bdr);
  border-radius: 3px; padding: 1px 6px;
  font-family: var(--mono); font-size: .82em; color: var(--amber);
}
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

/* ── Thinking blocks ── */
.thinking-block {
  margin: 2px 0 2px 4px;
  border-left: 2px solid var(--border2);
  padding: 3px 10px;
  font-size: 11px;
  font-family: var(--mono);
  color: var(--text3);
  cursor: pointer;
  user-select: none;
  transition: border-color .15s, opacity .15s;
  opacity: 0.6;
  line-height: 1.5;
}
.thinking-block:hover { border-left-color: var(--purple); opacity: 0.9; }
.thinking-block .tb-header {
  display: flex; align-items: baseline; gap: 6px;
}
.thinking-block .tb-arrow {
  font-size: 9px; color: var(--purple);
  transition: transform .15s; flex-shrink: 0;
  display: inline-block;
}
.thinking-block.open .tb-arrow { transform: rotate(90deg); }
.thinking-block .tb-preview { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.thinking-block .tb-body {
  display: none; white-space: pre-wrap; word-break: break-word;
  margin-top: 4px; color: var(--text3); line-height: 1.55;
}
.thinking-block.open .tb-body { display: block; }

.msg-copy {
  position: absolute; top: 30px; right: 10px;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: 4px; color: var(--text3); font-size: 10px;
  padding: 2px 8px; cursor: pointer; opacity: 0; pointer-events: none;
  transition: opacity .15s, color .12s, border-color .12s;
  font-family: var(--mono); user-select: none; z-index: 2;
}
.msg.agent:hover .msg-copy { opacity: 1; pointer-events: auto; }
.msg-copy:hover { color: var(--teal); border-color: var(--teal-dim); }
.msg-copy.copied { color: var(--green); border-color: var(--green); opacity: 1; }

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

.tool-group-summary { display: none; }
.tool-group.collapsible .tool-group-summary {
  display: flex; align-items: center; gap: 6px;
  padding: 2px 0 2px 8px; font-size: 11px; color: var(--text3);
  cursor: pointer; user-select: none; font-family: var(--mono);
  transition: color .12s;
}
.tool-group.collapsible .tool-group-summary:hover { color: var(--text2); }
.tgs-arrow { font-size: 9px; display: inline-block; transition: transform .15s; }
.tool-group.collapsed .tool-row,
.tool-group.collapsed .tool-more { display: none !important; }
.tool-group.collapsed .tgs-arrow { transform: rotate(-90deg); }

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

#elapsed-timer {
  font-size: 10px; color: var(--text3); font-family: var(--mono);
  display: none; letter-spacing: .03em;
}
#elapsed-timer.show { display: inline; }

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

/* ── Panels (shared base) ── */
.panel {
  position: absolute; top: 0; right: 0;
  width: min(320px, 90vw); height: 100%;
  background: var(--surface); border-left: 1px solid var(--border);
  transform: translateX(100%); transition: transform .2s ease;
  z-index: 20; display: flex; flex-direction: column;
}
.panel.open { transform: translateX(0); }
#tools-panel { width: min(360px, 92vw); }

/* ── File tree panel (left side) ── */
#files-panel {
  right: auto;
  left: 0;
  width: min(380px, 92vw);
  border-left: none;
  border-right: 1px solid var(--border);
  transform: translateX(-100%);
}
#files-panel.open { transform: translateX(0); }

/* ── File tree internals ── */
.ft-toolbar {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 10px; border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.ft-toolbar-title {
  flex: 1; font-family: var(--sans); font-weight: 600;
  font-size: 12px; color: var(--text2);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ft-icon-btn {
  width: 24px; height: 24px; border-radius: 4px;
  border: 1px solid var(--border); background: transparent;
  color: var(--text3); cursor: pointer; font-size: 12px;
  display: flex; align-items: center; justify-content: center;
  transition: border-color .1s, color .1s; flex-shrink: 0;
}
.ft-icon-btn:hover { border-color: var(--amber); color: var(--amber); }

.ft-search {
  margin: 7px 8px 5px; flex-shrink: 0;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 5px; color: var(--text); font-family: var(--mono);
  font-size: 12px; padding: 5px 10px; outline: none;
  width: calc(100% - 16px);
}
.ft-search:focus { border-color: var(--amber); }
.ft-search::placeholder { color: var(--text3); }

#ft-tree {
  flex: 1; overflow-y: auto; padding: 2px 0 6px;
  font-size: 12px; font-family: var(--mono);
}
#ft-tree::-webkit-scrollbar { width: 3px; }

.ft-msg {
  padding: 10px 14px; color: var(--text3);
  font-size: 11px; font-style: italic;
}

.ft-node {
  display: flex; align-items: center; gap: 5px;
  cursor: pointer; min-height: 22px; padding-right: 8px;
  transition: background .07s; white-space: nowrap; overflow: hidden;
}
.ft-node:hover { background: rgba(255,255,255,.03); }
.ft-node.ft-selected { background: rgba(232,162,69,.07); }
.ft-node.ft-loading-node { color: var(--text3); cursor: default; padding-left: 0; }

.ft-arrow {
  width: 10px; text-align: center; flex-shrink: 0;
  font-size: 7px; color: var(--text3);
  transition: transform .12s; display: inline-block;
}
.ft-node.ft-expanded > .ft-arrow { transform: rotate(90deg); }

.ft-name {
  flex: 1; overflow: hidden; text-overflow: ellipsis;
  color: var(--text2); user-select: none;
}
.ft-node.ft-dir > .ft-name { color: var(--text); font-weight: 500; }
.ft-node.ft-selected > .ft-name { color: var(--amber); }

.ft-size {
  flex-shrink: 0; color: var(--text3); font-size: 10px;
  min-width: 34px; text-align: right;
}

.ft-ext-badge {
  flex-shrink: 0; font-size: 9px; padding: 0 4px;
  border-radius: 2px; font-family: var(--mono);
  border: 1px solid; text-transform: uppercase;
  letter-spacing: .04em; line-height: 16px;
}
.ft-ext-py   { color: #4ec9b0; border-color: #1a3c34; }
.ft-ext-js   { color: #e8c45a; border-color: #504218; }
.ft-ext-ts   { color: #569cd6; border-color: #18304c; }
.ft-ext-html { color: #ce9178; border-color: #4c301c; }
.ft-ext-css  { color: #9d7dea; border-color: #362258; }
.ft-ext-md   { color: #8e8c88; border-color: #28282e; }
.ft-ext-json { color: #3ecfb2; border-color: #0c342e; }
.ft-ext-sh   { color: #3ec97a; border-color: #0c3224; }
.ft-ext-xml  { color: #ce9178; border-color: #4c301c; }
.ft-ext-yml  { color: #e8c45a; border-color: #504218; }
.ft-ext-txt  { color: var(--text3); border-color: var(--border); }
.ft-ext-def  { color: var(--text3); border-color: var(--border); }

.ft-divider { height: 1px; background: var(--border); flex-shrink: 0; }

#ft-preview {
  display: flex; flex-direction: column;
  flex-shrink: 0; min-height: 130px; max-height: 44%;
}

.ft-preview-hdr {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 8px 5px 10px; flex-shrink: 0;
  background: var(--surface2); border-bottom: 1px solid var(--border);
  min-height: 32px;
}
#ft-preview-path {
  flex: 1; font-size: 10px; color: var(--text2);
  font-family: var(--mono); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
.ft-preview-actions { display: flex; gap: 4px; flex-shrink: 0; }
.ft-preview-btn {
  font-family: var(--mono); font-size: 9px;
  padding: 2px 7px; border-radius: 3px;
  border: 1px solid var(--border); background: transparent;
  color: var(--text3); cursor: pointer;
  transition: border-color .1s, color .1s; white-space: nowrap;
}
.ft-preview-btn:hover { border-color: var(--amber); color: var(--amber); }

#ft-preview-content {
  flex: 1; overflow: auto;
  padding: 10px 12px;
  font-family: var(--mono); font-size: 11px;
  line-height: 1.52; background: var(--code-bg);
  color: var(--text2); white-space: pre;
  word-break: break-all; tab-size: 2;
}
.ft-preview-hint {
  color: var(--text3) !important;
  font-style: italic; white-space: normal !important;
  font-size: 11px;
}

/* ── Sessions / Tools panels ── */
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
      <span class="logo-sub">v6.6.1</span>
    </div>
    <div class="header-right">
      <span class="session-pill" id="session-pill"></span>
      <div class="conn-dot" id="conn-dot" title="Stream connection"></div>
      <button class="btn" onclick="UI.toggleFiles()">Files</button>
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
      <div class="hint">/ for commands &nbsp;&middot;&nbsp; &uarr;&darr; history</div>
    </div>
  </div>

  <button id="scroll-btn" onclick="Scroll.jump()" title="&#8595;">&#8595;</button>

  <div id="status-bar">
    <div class="dot" id="dot"></div>
    <span id="status-text">ready</span>
    <span id="elapsed-timer"></span>
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

<!-- ── File tree panel (opens LEFT) ──────────────────────────────────────── -->
<div class="panel" id="files-panel">
  <div class="ft-toolbar">
    <span class="ft-toolbar-title" id="ft-ws-label">Workspace</span>
    <button class="ft-icon-btn" onclick="FileTree.refresh()" title="Refresh tree">&#8635;</button>
    <button class="ft-icon-btn" onclick="UI.closeFiles()" title="Close">&times;</button>
  </div>
  <input id="ft-search" class="ft-search" placeholder="Filter files&hellip;"
         oninput="FileTree.filter(this.value)">
  <div id="ft-tree"><div class="ft-msg">Loading&hellip;</div></div>
  <div class="ft-divider"></div>
  <div id="ft-preview">
    <div class="ft-preview-hdr">
      <span id="ft-preview-path">No file selected</span>
      <div class="ft-preview-actions">
        <button class="ft-preview-btn" onclick="FileTree.copyPath()">copy path</button>
        <button class="ft-preview-btn" onclick="FileTree.askAgent()">ask agent</button>
      </div>
    </div>
    <div id="ft-preview-content" class="ft-preview-hint">click a file to preview</div>
  </div>
</div>

<!-- ── Sessions panel (opens RIGHT) ─────────────────────────────────────── -->
<div class="panel" id="sessions-panel">
  <div class="panel-hdr">
    <span>Sessions</span>
    <button class="btn" onclick="UI.closeSessions()">&#x2715;</button>
  </div>
  <div id="sessions-list"></div>
</div>

<!-- ── Tools panel (opens RIGHT) ─────────────────────────────────────────── -->
<div class="panel" id="tools-panel">
  <div class="panel-hdr">
    <span>Tools</span>
    <button class="btn" onclick="UI.closeTools()">&#x2715;</button>
  </div>
  <input id="tools-search" placeholder="Filter&hellip;" oninput="ToolsPanel.filter(this.value)">
  <div id="tools-list"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
<script>
'use strict';

// ── Auth token (injected server-side) ─────────────────────────────────────────
const _AGENT_TOKEN = '__AGENT_TOKEN__';

// Monkey-patch window.fetch to auto-inject the auth header on every call.
// This is the single point of change — no individual fetch() needs updating.
(function() {
  var _origFetch = window.fetch.bind(window);
  window.fetch = function(url, opts) {
    opts = Object.assign({}, opts);
    opts.headers = Object.assign({'X-Token': _AGENT_TOKEN}, opts.headers || {});
    return _origFetch(url, opts);
  };
})();

function _esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function _makeThinkingBlock(text) {
  var el = document.createElement('div');
  el.className = 'thinking-block';
  var firstLine = text.split('\n')[0].slice(0, 120);
  var multiline = text.length > firstLine.length || text.indexOf('\n') !== -1;
  var hdr = document.createElement('div');
  hdr.className = 'tb-header';
  var arrow = document.createElement('span');
  arrow.className = 'tb-arrow';
  arrow.textContent = '\u25b6';
  var preview = document.createElement('span');
  preview.className = 'tb-preview';
  preview.textContent = '\ud83d\udcad ' + firstLine + (multiline ? '\u2026' : '');
  hdr.appendChild(arrow);
  hdr.appendChild(preview);
  el.appendChild(hdr);
  var body = document.createElement('div');
  body.className = 'tb-body';
  body.textContent = text;
  el.appendChild(body);
  el.onclick = function() {
    var open = el.classList.toggle('open');
    arrow.textContent = open ? '\u25bc' : '\u25b6';
    preview.textContent = open ? '\ud83d\udcad thinking\u2026' : '\ud83d\udcad ' + firstLine + (multiline ? '\u2026' : '');
  };
  return el;
}

const MD = (() => {
  marked.use({
    gfm: true,
    breaks: true,
    renderer: {
      code(code, lang) {
        var langBadge = lang ? '<span class="lang">' + _esc(lang) + '</span>' : '';
        return '<pre>' + langBadge + '<code>' + _esc(code) + '</code></pre>\n';
      },
      link(href, title, text) {
        return '<a href="' + _esc(href || '') + '" target="_blank" rel="noopener"'
          + (title ? ' title="' + _esc(title) + '"' : '') + '>' + text + '</a>';
      },
      image(href, title, text) {
        return '<img src="' + _esc(href || '') + '" alt="' + _esc(text || '') + '"'
          + ' style="max-width:100%"'
          + (title ? ' title="' + _esc(title) + '"' : '') + '>';
      }
    }
  });
  var MAX_RENDER_BYTES = 400000;
  function render(md) {
    if (!md || !md.trim()) return '';
    if (md.length > MAX_RENDER_BYTES)
      return '<pre style="white-space:pre-wrap;word-break:break-word">' + _esc(md) + '</pre>';
    try { return marked.parse(md); }
    catch (_) { return '<pre style="white-space:pre-wrap">' + _esc(md) + '</pre>'; }
  }
  return { render };
})();

const Scroll = (() => {
  var pinned = true;
  var pane   = function(){ return document.getElementById('messages'); };
  var btn    = function(){ return document.getElementById('scroll-btn'); };
  function atBottom() {
    var el = pane();
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }
  function update() { btn().classList.toggle('show', !pinned && Agent.running()); }
  function maybe()  { if (pinned) pane().scrollTop = pane().scrollHeight; }
  function jump()   { pinned = true; pane().scrollTop = pane().scrollHeight; update(); }
  function onScroll() {
    pinned = atBottom() ? true : Agent.running() ? false : pinned;
    update();
  }
  pane().addEventListener('scroll', onScroll, { passive: true });
  return { maybe, jump, pin: function(){ pinned = true; }, update };
})();

const Status = (() => {
  return {
    set: function(msg, state) {
      document.getElementById('status-text').textContent = msg || '';
      document.getElementById('dot').className = 'dot ' + (state || '');
    },
    iter: function(v) {
      var el = document.getElementById('iter-badge');
      el.textContent = v || '';
      el.style.display = v ? '' : 'none';
    },
    mode: function(v) { document.getElementById('mode-badge').textContent = v || 'auto'; },
  };
})();

const ElapsedTimer = (() => {
  var _t = null, _start = 0;
  var el = function(){ return document.getElementById('elapsed-timer'); };
  function _fmt(ms) {
    var s = Math.floor(ms / 1000), m = Math.floor(s / 60);
    s = s % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }
  function start() {
    stop();
    _start = Date.now();
    el().classList.add('show');
    _t = setInterval(function(){ el().textContent = _fmt(Date.now() - _start); }, 500);
  }
  function stop() {
    if (_t) { clearInterval(_t); _t = null; }
    el().classList.remove('show');
    el().textContent = '';
  }
  return { start, stop };
})();

const ConnDot = (() => {
  var el = function(){ return document.getElementById('conn-dot'); };
  return {
    ok:  function(){ el().className = 'conn-dot ok';  el().title = 'Stream connected'; },
    err: function(){ el().className = 'conn-dot err'; el().title = 'Stream disconnected'; },
    try: function(){ el().className = 'conn-dot try'; el().title = 'Reconnecting\u2026'; },
  };
})();

const InputHistory = (() => {
  var _hist = [], _idx = -1, _draft = '';
  function _set(inp, val) {
    inp.value = val;
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 140) + 'px';
    setTimeout(function(){ inp.selectionStart = inp.selectionEnd = inp.value.length; }, 0);
  }
  function push(text) {
    if (!text || (_hist.length && _hist[_hist.length - 1] === text)) return;
    _hist.push(text);
    if (_hist.length > 200) _hist.shift();
    _idx = -1;
  }
  function up(inp) {
    if (!_hist.length) return;
    if (_idx === -1) { _draft = inp.value; _idx = _hist.length; }
    if (_idx > 0) _set(inp, _hist[--_idx]);
  }
  function down(inp) {
    if (_idx === -1) return;
    _idx++;
    _set(inp, _idx >= _hist.length ? (_idx = -1, _draft) : _hist[_idx]);
  }
  function reset() { _idx = -1; }
  return { push, up, down, reset };
})();

const Replay = (() => {
  function _addCopyBtn(wrap, rawText) {
    var btn = document.createElement('button');
    btn.className = 'msg-copy';
    btn.textContent = 'copy';
    btn.title = 'Copy response';
    btn.onclick = function(e) {
      e.stopPropagation();
      navigator.clipboard.writeText(rawText || wrap.querySelector('.msg-body').innerText || '').then(function() {
        btn.textContent = 'copied!'; btn.classList.add('copied');
        setTimeout(function(){ btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1500);
      }).catch(function(){});
    };
    wrap.appendChild(btn);
  }

  function _applyEvents(events) {
    var tokenBuf = '', toolGroupItems = [];
    var pane     = document.getElementById('messages');

    function flushTokens() {
      if (!tokenBuf.trim()) { tokenBuf = ''; return; }
      document.getElementById('empty') && document.getElementById('empty').remove();
      var wrap = document.createElement('div');
      wrap.className = 'msg agent';
      wrap.innerHTML = '<div class="msg-label">Agent</div><div class="msg-body"></div>';
      var body  = wrap.querySelector('.msg-body');
      body.innerHTML = MD.render(tokenBuf) || _esc(tokenBuf);
      _addCopyBtn(wrap, tokenBuf);
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
        var preview = t.args ? (t.args.slice(0,50) + (t.args.length > 50 ? '\u2026' : '')) : '';
        row.innerHTML = '<span class="tr-icon">'+(ok === null ? '\u25cc' : (ok ? '\u2713' : '\u2717'))+'</span><span class="tr-name">'+_esc(t.name)+'</span>'+(preview ? '<span class="tr-args">('+_esc(preview)+')</span>' : '');
        group.appendChild(row);
      }
      _finaliseToolGroup(group, toolGroupItems.length);
      pane.appendChild(group);
      toolGroupItems = [];
    }

    function _sysMsg(text, variant) {
      var d = document.createElement('div');
      d.className = 'msg sys ' + (variant || '');
      d.innerHTML = '<div class="msg-label"></div><div class="msg-body"></div>';
      d.querySelector('.msg-body').textContent = text;
      pane.appendChild(d);
    }

    var lastWasToken = false;
    for (var ei = 0; ei < events.length; ei++) {
      var kind = events[ei][0], payload = events[ei][1];
      if (kind === 'token') {
        if (!lastWasToken) flushTools();
        tokenBuf += payload; lastWasToken = true;
      } else if (kind === 'thinking') {
        flushTokens(); flushTools(); lastWasToken = false;
        document.getElementById('empty') && document.getElementById('empty').remove();
        pane.appendChild(_makeThinkingBlock(payload));
      } else if (kind === 'tool') {
        if (lastWasToken) flushTokens();
        lastWasToken = false;
        if (payload.startsWith('\u2713') || payload.startsWith('\u2717')) {
          var name2 = payload.slice(1).trim();
          for (var ri = toolGroupItems.length - 1; ri >= 0; ri--) {
            if (toolGroupItems[ri].name === name2 && toolGroupItems[ri].ok === null) {
              toolGroupItems[ri].ok = payload.startsWith('\u2713'); break;
            }
          }
        } else {
          flushTokens();
          var m2 = payload.match(/^([^(]+)\(?([\s\S]*?)\)?$/);
          toolGroupItems.push({ name: m2 ? m2[1].trim() : payload, args: m2 ? m2[2].trim() : '', ok: null });
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

  function run(events) {
    if (!events || !events.length) return;
    var banner = document.getElementById('replay-banner');
    banner.textContent = '\u21a9 restored ' + events.length + ' events from this session';
    banner.classList.add('show');
    setTimeout(function(){ banner.classList.remove('show'); }, 3000);
    var CHUNK = 200, offset = 0;
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

function _finaliseToolGroup(group, totalCount) {
  if (totalCount < 3) return;
  var okCount   = group.querySelectorAll('.tool-row[data-s="ok"]').length;
  var failCount = group.querySelectorAll('.tool-row[data-s="fail"]').length;
  function _label(collapsed) {
    if (collapsed) return '\u25b6 ' + totalCount + ' tool call' + (totalCount !== 1 ? 's' : '') + ' \u2014 click to expand';
    var parts = [totalCount + ' tool call' + (totalCount !== 1 ? 's' : '')];
    if (okCount)   parts.push(okCount + ' ok');
    if (failCount) parts.push(failCount + ' failed');
    return '\u25bc ' + parts.join(', ') + ' \u2014 click to collapse';
  }
  var summary = document.createElement('div');
  summary.className   = 'tool-group-summary';
  summary.textContent = _label(false);
  group.insertBefore(summary, group.firstChild);
  group.classList.add('collapsible');
  summary.onclick = function() {
    var c = group.classList.toggle('collapsed');
    summary.textContent = _label(c);
  };
}

const Messages = (() => {
  var pane = function(){ return document.getElementById('messages'); };
  function hideEmpty() { var e = document.getElementById('empty'); if (e) e.remove(); }
  function add(role, html, extra, asHtml) {
    hideEmpty();
    Tools.endGroup();
    var d = document.createElement('div');
    d.className = 'msg ' + role + (extra ? ' ' + extra : '');
    var labels = { user: 'You', agent: 'Agent' };
    d.innerHTML = '<div class="msg-label">'+(labels[role]||'')+'</div><div class="msg-body"></div>';
    pane().appendChild(d);
    var body = d.querySelector('.msg-body');
    if (html) { if (asHtml) body.innerHTML = html; else body.textContent = html; }
    Scroll.maybe();
    return body;
  }
  function sys(text, variant) { add('sys', text, variant || ''); }
  return { add, sys, pane, hideEmpty };
})();

const Stream = (() => {
  var el = null, buf = '', _rafPending = false, _cursor = null;

  function _attachCopyBtn(wrap, rawText) {
    var btn = document.createElement('button');
    btn.className = 'msg-copy'; btn.textContent = 'copy'; btn.title = 'Copy response';
    btn.onclick = function(e) {
      e.stopPropagation();
      var copy = function() {
        btn.textContent = 'copied!'; btn.classList.add('copied');
        setTimeout(function(){ btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1500);
      };
      navigator.clipboard.writeText(rawText).then(copy).catch(function() {
        var ta = document.createElement('textarea');
        ta.value = rawText; document.body.appendChild(ta); ta.select();
        document.execCommand('copy'); ta.remove(); copy();
      });
    };
    wrap.appendChild(btn);
  }

  function _renderFinal(body, rawBuf) {
    requestAnimationFrame(function() {
      var wrap = body.closest('.msg');
      if (!rawBuf.trim()) {
        if (wrap) wrap.remove();
      } else {
        body.innerHTML = MD.render(rawBuf) || _esc(rawBuf);
        if (wrap) _attachCopyBtn(wrap, rawBuf);
      }
      Scroll.maybe();
    });
  }

  function start() {
    buf = ''; el = Messages.add('agent', '');
    _cursor = document.createElement('span');
    _cursor.className = 'cursor';
    el.appendChild(_cursor);
  }

  function flush() {
    _rafPending = false;
    if (!el) return;
    el.innerHTML = MD.render(buf) || _esc(buf);
    el.appendChild(_cursor);
    Scroll.maybe();
  }

  function token(tok) {
    buf += tok;
    if (!_rafPending) {
      _rafPending = true;
      requestAnimationFrame(flush);
    }
  }

  function finalize() {
    _rafPending = false;
    if (!el) return;
    var body = el, rawBuf = buf;
    el = null; buf = ''; _cursor = null;
    _renderFinal(body, rawBuf);
  }

  function reset() {
    _rafPending = false;
    if (!el) return;
    var body = el, rawBuf = buf;
    el = null; buf = ''; _cursor = null;
    if (rawBuf.trim()) _renderFinal(body, rawBuf);
    else { var wrap = body.closest('.msg'); if (wrap) wrap.remove(); }
  }

  function active() { return el !== null; }
  return { start, token, finalize, reset, active };
})();

const Tools = (() => {
  var MAX = 12, group = null, count = 0, pending = new Map();

  function ensureGroup() {
    if (group) return group;
    var e = document.getElementById('empty'); if (e) e.remove();
    group = document.createElement('div');
    group.className = 'tool-group';
    Messages.pane().appendChild(group);
    count = 0;
    return group;
  }

  function endGroup() {
    if (group && count > 0) _finaliseToolGroup(group, count);
    group = null; count = 0;
  }

  function makeRow(name, args) {
    var row = document.createElement('div');
    row.className = 'tool-row'; row.dataset.s = 'pending';
    var preview = args ? (args.slice(0, 50) + (args.length > 50 ? '\u2026' : '')) : '';
    row.innerHTML = '<span class="tr-icon">\u25cc</span><span class="tr-name">'+_esc(name)+'</span>'+(preview ? '<span class="tr-args">('+_esc(preview)+')</span>' : '');
    if (!pending.has(name)) pending.set(name, []);
    pending.get(name).push(row);
    return row;
  }

  function call(name, args) {
    var g = ensureGroup(); count++;
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

const Whisper = (() => {
  function queue(text) {
    if (!Agent.running()) { Messages.sys('Agent is not running \u2014 whisper ignored.', 'warn'); return; }
    fetch('/whisper', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text }),
    }).catch(function(){});
  }
  return { queue };
})();

const Agent = (() => {
  var state = 'idle', sessionId = null, requestId = null, _hadContent = false;

  function setSession(id) {
    sessionId = id;
    var pill = document.getElementById('session-pill');
    if (id) { pill.textContent = id.slice(-8); pill.style.display = ''; }
    else    { pill.style.display = 'none'; }
  }

  function getSession() { return sessionId; }
  function running()    { return state !== 'idle'; }

  function lockUI(on) {
    var inp = document.getElementById('msg-input');
    var btn = document.getElementById('send-btn');
    inp.disabled    = false;
    inp.placeholder = on ? 'Whisper to agent\u2026 (nudge mid-run, Enter to send)' : 'Message or /command\u2026';
    btn.textContent = on ? 'Stop' : 'Send';
    btn.className   = on ? 'stop' : '';
  }

  // FIX: removed duplicate Scroll.update() — Scroll.update() is called once
  // below after lockUI, not additionally inside lockUI.
  function transition(next) {
    state = next;
    lockUI(next !== 'idle');
    if (next === 'idle') Scroll.pin();
    Scroll.update();
    if (next !== 'idle') ElapsedTimer.start(); else ElapsedTimer.stop();
  }

  function handleEvent(evt) {
    if (evt.type === 'connect') {
      var d = evt.data || {};
      if (d.session) setSession(d.session);
      if (d.mode)    Status.mode(d.mode);
      if (d.history && d.history.length && !document.getElementById('messages').querySelectorAll('.msg').length)
        Replay.run(d.history);
      if (d.state === 'running' && state === 'idle') {
        transition('running'); Status.set('running\u2026', 'run');
      } else if (d.state === 'waiting' && state === 'idle') {
        transition('waiting');
        Status.set('waiting \u2014 scheduled resume\u2026', 'wait');
        Messages.sys('\u23f0 Session waiting \u2014 will resume automatically', 'warn');
      } else if (d.state === 'idle' && state !== 'idle') {
        Stream.finalize(); Tools.endGroup(); transition('idle');
        Status.set('done', 'done'); Status.iter('');
        Messages.sys('\u2713 task finished', 'ok');
        document.getElementById('msg-input').focus();
      }
      return;
    }

    if (state === 'waiting' && evt.type !== 'connect') {
      transition('running'); Tools.reset(); Stream.reset();
    }

    switch (evt.type) {
      case 'token':
        Tools.endGroup();
        if (!Stream.active()) Stream.start();
        Stream.token(evt.data);
        _hadContent = true;
        break;

      case 'thinking': {
        Stream.finalize();
        Tools.endGroup();
        Messages.hideEmpty();
        Messages.pane().appendChild(_makeThinkingBlock(evt.data));
        Scroll.maybe();
        _hadContent = true;
        break;
      }

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
          Tools.call(m ? m[1].trim() : t, m ? m[2].trim() : '');
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
          Stream.finalize(); Tools.endGroup();
          if (_hadContent) Messages.sys('\u2014 iteration '+mi[1]+'/'+mi[2]+' \u2014', 'info');
          _hadContent = false;
          Status.iter(mi[1]+'/'+mi[2]);
        }
        break;
      }

      case 'done':
        onDone(evt.data);
        break;

      case 'error':
        Stream.finalize(); Tools.endGroup();
        Messages.sys('\u26a0 ' + evt.data, 'err');
        transition('idle'); Status.set('error', 'err'); Status.iter('');
        break;
    }
  }

  function onDone(reason) {
    Stream.finalize(); Tools.endGroup(); Status.iter('');
    var isWait = String(reason || '').startsWith('waiting');
    var isErr  = reason === 'error' || reason === 'stopped';
    if (isWait) {
      transition('waiting');
      Status.set('waiting \u2014 will resume automatically', 'wait');
      Messages.sys('\u23f8 ' + reason, 'warn');
    } else {
      transition('idle');
      Status.set(reason || 'done', isErr ? 'err' : 'done');
      if (reason === 'stopped') Messages.sys('\u2014 stopped \u2014', 'warn');
      else if (!isErr) Messages.sys('\u2713 ' + ((reason||'').replace(/\s*\u2713\s*$/,'').trim() || 'task finished'), 'ok');
      document.getElementById('msg-input').focus();
    }
  }

  async function send(text) {
    if (state !== 'idle') return;
    Messages.add('user', text);
    Scroll.pin(); transition('running'); Status.set('thinking\u2026', 'run');
    Tools.reset(); _hadContent = false;
    requestId = 'r' + Date.now() + Math.random().toString(36).slice(2, 6);
    try {
      var resp = await fetch('/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId, request_id: requestId }),
      });
      if (!resp.ok) {
        var err = await resp.json().catch(function(){ return { error: 'HTTP ' + resp.status }; });
        throw new Error(err.error || 'HTTP ' + resp.status);
      }
    } catch (err) {
      Stream.finalize(); Tools.endGroup();
      Messages.sys('Failed to send: ' + err.message, 'err');
      transition('idle'); Status.set('error', 'err'); Status.iter('');
    }
  }

  function stop() {
    if (state === 'waiting') {
      transition('idle'); Status.set('wait dismissed', '');
      Messages.sys('\u2014 wait dismissed (session saved) \u2014', 'warn'); return;
    }
    if (state !== 'running') return;
    if (requestId) {
      fetch('/stop', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: requestId }),
      }).catch(function(){});
      requestId = null;
    }
    Stream.reset(); Tools.endGroup();
    Messages.sys('\u2014 stopped \u2014', 'warn');
    transition('idle'); Status.set('stopped', ''); Status.iter('');
  }

  async function newSession() {
    if (state !== 'idle') { stop(); return; }
    try { await fetch('/new', { method: 'POST' }); } catch (_) {}
    sessionId = null; requestId = null; _hadContent = false;
    setSession(null); Tools.reset(); Stream.reset();
    document.getElementById('messages').innerHTML =
      '<div id="replay-banner"></div><div id="empty"><div class="empty-glyph">&#955;_</div><p>Send a message to start</p><div class="hint">/ for commands &nbsp;&middot;&nbsp; &uarr;&darr; history</div></div>';
    Status.set('ready', ''); Status.iter('');
    document.getElementById('msg-input').focus();
  }

  function sendOrStop() {
    if (state !== 'idle') {
      var inp  = document.getElementById('msg-input');
      var text = inp.value.trim();
      if (text) {
        inp.value = ''; inp.style.height = 'auto';
        Whisper.queue(text);
        Messages.sys('\ud83d\udcac whisper sent \u2014 agent will receive it on next iteration', 'ok');
      } else {
        stop();
      }
    } else {
      _triggerSend();
    }
  }

  function _triggerSend() {
    var inp  = document.getElementById('msg-input');
    var text = inp.value.trim();
    if (!text) return;
    inp.value = ''; inp.style.height = 'auto';
    Palette.close(); InputHistory.reset();
    if (text.startsWith('/')) { Messages.add('user', text); SlashCmds.run(text); }
    else { InputHistory.push(text); send(text); }
  }

  return { sendOrStop, newSession, running, getSession, setSession, _handle: handleEvent };
})();

(function initStream() {
  var es = null, retryMs = 1000;
  function connect() {
    ConnDot.try();
    es = new EventSource('/stream?token=' + encodeURIComponent(_AGENT_TOKEN));
    es.onopen  = function() { ConnDot.ok(); retryMs = 1000; };
    es.onmessage = function(e) {
      if (!e.data || e.data === '[DONE]') return;
      try { Agent._handle(JSON.parse(e.data)); } catch (_) {}
    };
    es.onerror = function() {
      es.close(); ConnDot.err();
      setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, 30000);
    };
  }
  connect();
})();

// =============================================================================
// FILE TREE
// =============================================================================
const FileTree = (() => {
  var _expanded  = {};
  var _children  = {};
  var _loading   = {};
  var _selected  = null;
  var _filter    = '';
  var _autoTimer = null;
  var _lastMtime = 0;  // OPT: skip refresh when workspace unchanged

  var _EXT_MAP = {
    '.py':'.py', '.pyw':'.py',
    '.js':'.js', '.mjs':'.js', '.jsx':'.js',
    '.ts':'.ts', '.tsx':'.ts',
    '.html':'.html', '.htm':'.html',
    '.css':'.css', '.scss':'.css', '.sass':'.css', '.less':'.css',
    '.md':'.md', '.markdown':'.md',
    '.json':'.json', '.jsonl':'.json',
    '.sh':'.sh', '.bash':'.sh', '.zsh':'.sh', '.fish':'.sh',
    '.xml':'.xml',
    '.yml':'.yml', '.yaml':'.yml',
    '.txt':'.txt',
  };

  function _extKey(ext) {
    var k = _EXT_MAP[ext];
    return k ? k.slice(1) : 'def';
  }

  function _fmtSize(bytes) {
    if (!bytes) return '';
    if (bytes < 1024)    return bytes + 'B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + 'K';
    return (bytes / 1048576).toFixed(1) + 'M';
  }

  async function _fetchDir(path) {
    _loading[path] = true;
    try {
      var r = await fetch('/filetree?path=' + encodeURIComponent(path));
      var d = await r.json();
      if (d.error) throw new Error(d.error);
      _children[path] = d.entries || [];
    } finally {
      delete _loading[path];
    }
    return _children[path];
  }

  async function _fetchFile(path) {
    var r = await fetch('/fileread?path=' + encodeURIComponent(path));
    return await r.json();
  }

  async function _fetchMtime() {
    try {
      var r = await fetch('/workspace/mtime');
      var d = await r.json();
      return d.mtime || 0;
    } catch(_) { return 0; }
  }

  function _flatten(path, depth, out) {
    if (_loading[path] && !_children[path]) {
      out.push({ kind: 'loading', depth: depth });
      return;
    }
    var entries = _children[path] || [];
    var q = _filter.toLowerCase();
    var filtered = q ? entries.filter(function(e) {
      return e.name.toLowerCase().indexOf(q) !== -1 || e.type === 'dir';
    }) : entries;

    for (var i = 0; i < filtered.length; i++) {
      var e = filtered[i];
      out.push({ kind: 'entry', depth: depth, entry: e });
      if (e.type === 'dir' && _expanded[e.path]) {
        _flatten(e.path, depth + 1, out);
      }
    }
  }

  function _render() {
    var el = document.getElementById('ft-tree');
    if (!el) return;
    var panel = document.getElementById('files-panel');
    if (!panel || !panel.classList.contains('open')) return;

    var nodes = [];
    _flatten('.', 0, nodes);

    if (!nodes.length && _loading['.']) {
      el.innerHTML = '<div class="ft-msg">Loading workspace\u2026</div>';
      return;
    }
    if (!nodes.length && !_children['.']) {
      el.innerHTML = '<div class="ft-msg">Empty workspace</div>';
      return;
    }
    if (!nodes.length) {
      el.innerHTML = '<div class="ft-msg">No files' + (_filter ? ' matching \u201c' + _esc(_filter) + '\u201d' : '') + '</div>';
      return;
    }

    var frag = document.createDocumentFragment();
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      var row = document.createElement('div');

      if (n.kind === 'loading') {
        row.className = 'ft-node ft-loading-node';
        row.style.paddingLeft = (n.depth * 14 + 24) + 'px';
        row.innerHTML = '<span style="color:var(--text3);font-style:italic;font-size:11px">\u2026</span>';
        frag.appendChild(row);
        continue;
      }

      var e = n.entry;
      var isDir  = e.type === 'dir';
      var isSel  = _selected === e.path;
      var isExp  = !!_expanded[e.path];

      row.className = 'ft-node' +
        (isDir ? ' ft-dir' : ' ft-file') +
        (isExp ? ' ft-expanded' : '') +
        (isSel ? ' ft-selected' : '');
      row.style.paddingLeft = (n.depth * 14 + 6) + 'px';

      var arrowHtml = isDir
        ? '<span class="ft-arrow">\u25b6</span>'
        : '<span class="ft-arrow" style="visibility:hidden">\u25b6</span>';

      var badgeHtml = '';
      if (!isDir && e.ext) {
        var cls = 'ft-ext-' + _extKey(e.ext);
        badgeHtml = '<span class="ft-ext-badge ' + cls + '">' + _esc(e.ext.slice(1)) + '</span>';
      }

      var sizeHtml = (!isDir && e.size)
        ? '<span class="ft-size">' + _fmtSize(e.size) + '</span>'
        : '';

      row.innerHTML = arrowHtml +
        '<span class="ft-name" title="' + _esc(e.path) + '">' + _esc(e.name) + '</span>' +
        badgeHtml + sizeHtml;

      (function(entry) {
        row.onclick = function(ev) {
          ev.stopPropagation();
          if (entry.type === 'dir') _toggleDir(entry.path);
          else _openFile(entry.path);
        };
      })(e);

      frag.appendChild(row);
    }
    el.innerHTML = '';
    el.appendChild(frag);
  }

  async function _toggleDir(path) {
    if (_expanded[path]) {
      delete _expanded[path];
      _render();
    } else {
      _expanded[path] = true;
      _render();
      if (!_children[path] && !_loading[path]) {
        try {
          await _fetchDir(path);
        } catch(err) {
          _children[path] = [];
          delete _expanded[path];
        }
      }
      _render();
    }
  }

  async function _openFile(path) {
    _selected = path;
    _render();
    var pathEl    = document.getElementById('ft-preview-path');
    var contentEl = document.getElementById('ft-preview-content');
    if (pathEl)    pathEl.textContent = path;
    if (contentEl) { contentEl.className = ''; contentEl.textContent = 'Loading\u2026'; }

    try {
      var d = await _fetchFile(path);
      if (!contentEl) return;
      if (d.error) {
        contentEl.textContent = '\u26a0 ' + d.error;
        contentEl.className = 'ft-preview-hint';
      } else if (d.binary) {
        contentEl.textContent = '[binary file \u2014 ' + _fmtSize(d.size) + ']';
        contentEl.className = 'ft-preview-hint';
      } else {
        contentEl.className = '';
        contentEl.textContent = (d.content !== undefined && d.content !== null)
          ? (d.content || '(empty file)')
          : '(empty file)';
        if (d.truncated) {
          contentEl.textContent += '\n\n[\u2026 preview truncated at 100\u00a0KB \u2026]';
        }
      }
    } catch(err) {
      if (contentEl) {
        contentEl.textContent = 'Error: ' + err.message;
        contentEl.className = 'ft-preview-hint';
      }
    }
  }

  async function load() {
    _expanded  = {};
    _children  = {};
    _loading   = {};
    _selected  = null;
    _lastMtime = 0;
    var el = document.getElementById('ft-tree');
    if (el) el.innerHTML = '<div class="ft-msg">Loading workspace\u2026</div>';
    var pathEl = document.getElementById('ft-preview-path');
    var contEl = document.getElementById('ft-preview-content');
    if (pathEl) pathEl.textContent = 'No file selected';
    if (contEl) { contEl.textContent = 'click a file to preview'; contEl.className = 'ft-preview-hint'; }

    var lbl = document.getElementById('ft-ws-label');
    try {
      var s = await fetch('/status').then(function(r){ return r.json(); });
      if (lbl && s.workspace) {
        var parts = s.workspace.replace(/\\/g, '/').split('/');
        lbl.textContent = parts[parts.length - 1] || 'Workspace';
        lbl.title = s.workspace;
      }
    } catch(_) {}

    try {
      await _fetchDir('.');
      _lastMtime = await _fetchMtime();
      _render();
    } catch(err) {
      if (el) el.innerHTML = '<div class="ft-msg" style="color:var(--red)">\u26a0 ' + _esc(String(err.message || err)) + '</div>';
    }
    startAutoRefresh();
  }

  async function refresh() {
    var paths = ['.'];
    Object.keys(_expanded).forEach(function(p) { if (paths.indexOf(p) === -1) paths.push(p); });
    await Promise.all(paths.map(function(p) {
      return _fetchDir(p).catch(function(){});
    }));
    _render();
  }

  function filter(text) {
    _filter = text;
    _render();
  }

  function copyPath() {
    if (!_selected) return;
    var pathEl = document.getElementById('ft-preview-path');
    navigator.clipboard.writeText(_selected).then(function() {
      if (pathEl) {
        var orig = pathEl.textContent;
        pathEl.textContent = '\u2713 copied!';
        setTimeout(function() { pathEl.textContent = orig; }, 1200);
      }
    }).catch(function() {
      var ta = document.createElement('textarea');
      ta.value = _selected;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    });
  }

  function askAgent() {
    if (!_selected) return;
    var inp = document.getElementById('msg-input');
    if (inp) {
      inp.value = _selected;
      inp.focus();
      inp.dispatchEvent(new Event('input'));
    }
    UI.closeFiles();
  }

  function startAutoRefresh() {
    stopAutoRefresh();
    // OPT: only fetch when workspace mtime has changed, avoiding needless I/O.
    _autoTimer = setInterval(async function() {
      if (typeof Agent !== 'undefined' && Agent.running()) {
        var mtime = await _fetchMtime();
        if (mtime !== _lastMtime) {
          _lastMtime = mtime;
          refresh();
        }
      }
    }, 2500);
  }

  function stopAutoRefresh() {
    if (_autoTimer) { clearInterval(_autoTimer); _autoTimer = null; }
  }

  return { load, refresh, filter, copyPath, askAgent, startAutoRefresh, stopAutoRefresh };
})();

// =============================================================================
// SLASH COMMANDS
// =============================================================================
const CMDS = [
  { cmd: '/help',     desc: 'Show commands' },
  { cmd: '/new',      desc: 'Start a fresh session' },
  { cmd: '/files',    desc: 'Browse workspace files' },
  { cmd: '/sessions', desc: 'Browse history' },
  { cmd: '/tools',    desc: 'View all tools' },
  { cmd: '/status',   desc: 'Connection info' },
  { cmd: '/mode',     desc: 'Set permission mode', arg: '<auto|normal|manual>' },
  { cmd: '/soul',     desc: 'Agent personality' },
  { cmd: '/todo',     desc: 'Session todos' },
  { cmd: '/plan',     desc: 'Active plan' },
  { cmd: '/session',  desc: 'Current session ID' },
  { cmd: '/whisper',  desc: 'Inject a nudge to the running agent', arg: '<message>' },
];

const HELP_TEXT = 'Slash commands:\n  /help               This text\n  /new                Fresh session\n  /files              Browse workspace files\n  /sessions           Browse sessions\n  /tools              Tool list + MCP\n  /status             Config info\n  /mode auto|normal|manual\n  /soul               Agent personality\n  /todo               Current todos\n  /plan               Current plan\n  /session            Session ID\n  /whisper <text>     Inject mid-run nudge\n\nKeyboard shortcuts:\n  \u2191 / \u2193             Cycle input history\n  Ctrl+L           New session\n  Enter            Send\n  Shift+Enter      New line';

const SlashCmds = (() => {
  async function run(text) {
    var parts = text.trim().split(/\s+/);
    var cmd   = parts[0].toLowerCase();
    var arg   = parts.slice(1).join(' ');
    var sys   = Messages.sys.bind(Messages);

    switch (cmd) {
      case '/help':     sys(HELP_TEXT, 'info'); break;
      case '/new':      Agent.newSession(); break;
      case '/files':    UI.toggleFiles(); break;
      case '/sessions': UI.toggleSessions(); break;
      case '/tools':    UI.toggleTools(); break;
      case '/session':
        sys(Agent.getSession() ? 'Session: ' + Agent.getSession() : 'No active session.', 'info'); break;
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
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: arg }),
          }).then(function(r){ return r.json(); });
          if (r2.ok) { Status.mode(arg); sys('Mode \u2192 ' + arg, 'ok'); }
          else       { sys('Invalid mode \''+arg+'\'. Use: auto, normal, manual', 'err'); }
        } catch (e) { sys('Failed: ' + e.message, 'err'); }
        break;
      }
      case '/whisper': {
        if (!arg) { sys('Usage: /whisper <message>', 'warn'); break; }
        Whisper.queue(arg);
        sys('Whisper queued \u2192 agent will receive it on next iteration.', 'ok');
        break;
      }
      case '/todo':
      case '/plan': {
        var sid = Agent.getSession();
        if (!sid) { sys('No active session.', 'warn'); break; }
        try {
          var r3 = await fetch('/cmd', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
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

const Palette = (() => {
  var idx = -1;
  var pal = function(){ return document.getElementById('palette'); };

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
    inp.value = (cmd === '/mode' || cmd === '/whisper') ? cmd + ' ' : cmd;
    close(); inp.focus();
  }

  function onInput(el) {
    autoResize(el);
    InputHistory.reset();
    if (el.value.startsWith('/')) build(el.value); else close();
  }

  function onKey(e) {
    var inp = document.getElementById('msg-input');
    var p   = pal();
    if (p.classList.contains('open')) {
      var items = p.querySelectorAll('.pi');
      if (e.key === 'ArrowDown') { e.preventDefault(); idx = Math.min(idx+1, items.length-1); items.forEach(function(el,i){ el.classList.toggle('sel', i===idx); }); return; }
      if (e.key === 'ArrowUp')   { e.preventDefault(); idx = Math.max(idx-1, 0);              items.forEach(function(el,i){ el.classList.toggle('sel', i===idx); }); return; }
      if (e.key === 'Tab' || (e.key === 'Enter' && idx >= 0)) { e.preventDefault(); (idx >= 0 ? items[idx] : items[0]) && (idx >= 0 ? items[idx] : items[0]).click(); return; }
      if (e.key === 'Escape') { close(); return; }
    }
    if (e.key === 'ArrowUp' && !p.classList.contains('open')) {
      if (inp.value.slice(0, inp.selectionStart).split('\n').length <= 1) { e.preventDefault(); InputHistory.up(inp); return; }
    }
    if (e.key === 'ArrowDown' && !p.classList.contains('open')) {
      if (inp.value.slice(inp.selectionEnd).split('\n').length <= 1) { e.preventDefault(); InputHistory.down(inp); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); Agent.sendOrStop(); }
  }

  return { onInput, onKey, close };
})();

function autoResize(el) {
  requestAnimationFrame(function() {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  });
}

document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'l') { e.preventDefault(); Agent.newSession(); }
});

const SessionsPanel = (() => {
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
          Agent.setSession(s.id); UI.closeSessions();
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

const ToolsPanel = (() => {
  var data = null;

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
    var qlo = q.toLowerCase();
    var all = Object.assign({}, data.builtin, data.mcp || {});
    var shown = false;
    Object.entries(all).forEach(function(entry) {
      var cat = entry[0], tools = entry[1];
      var filtered = tools.filter(function(t){ return !qlo || t.name.includes(qlo) || t.description.toLowerCase().includes(qlo); });
      if (!filtered.length) return;
      shown = true;
      var catEl = document.createElement('div');
      catEl.className = 'tool-cat' + (cat.startsWith('MCP') ? ' mcp-cat' : '');
      catEl.innerHTML = '<div class="tool-cat-hdr">'+_esc(cat)+'</div>';
      filtered.forEach(function(t) {
        var params = t.params.length ? t.params.map(function(p){ return t.required.includes(p) ? p : '['+p+']'; }).join(', ') : '';
        var entry2 = document.createElement('div');
        entry2.className = 'tool-entry';
        entry2.innerHTML = '<div class="te-name">'+_esc(t.name)+'<span class="te-params">'+(params ? '('+_esc(params)+')' : '')+'</span></div>'+(t.description ? '<div class="te-desc">'+_esc(t.description)+'</div>' : '');
        catEl.appendChild(entry2);
      });
      list.appendChild(catEl);
    });
    if (!shown) list.innerHTML = '<div style="padding:10px;color:var(--text3);font-size:11px">No matching tools</div>';
  }

  function filter(q) { if (data) render(q); }
  return { load, filter };
})();

// =============================================================================
// UI — panel orchestration
// =============================================================================
const UI = (() => {
  var overlay = function(){ return document.getElementById('overlay'); };
  function openPanel(id)  { document.getElementById(id).classList.add('open');    overlay().classList.add('show'); }
  function closePanel(id) { document.getElementById(id).classList.remove('open'); overlay().classList.remove('show'); }
  function isOpen(id)     { return document.getElementById(id).classList.contains('open'); }

  async function toggleFiles() {
    if (isOpen('files-panel')) { closeFiles(); return; }
    closeSessions(); closeTools();
    openPanel('files-panel');
    await FileTree.load();
  }
  async function toggleSessions() {
    if (isOpen('sessions-panel')) { closeSessions(); return; }
    closeTools(); closeFiles(); await SessionsPanel.load(); openPanel('sessions-panel');
  }
  async function toggleTools() {
    if (isOpen('tools-panel')) { closeTools(); return; }
    closeSessions(); closeFiles(); await ToolsPanel.load(); openPanel('tools-panel');
  }
  function closeFiles()    { closePanel('files-panel'); FileTree.stopAutoRefresh(); }
  function closeSessions() { closePanel('sessions-panel'); }
  function closeTools()    { closePanel('tools-panel'); }
  function closeAll()      { closeFiles(); closeSessions(); closeTools(); }

  return { toggleFiles, closeFiles, toggleSessions, closeSessions, toggleTools, closeTools, closeAll };
})();

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
    _running:      set = set()
    _running_lock = threading.Lock()

    def _run_wake(sid: str):
        global _current_session_id
        with _running_lock:
            _running.add(sid)

        last_status = [""]

        def _do_run():
            global _current_session_id
            _set_agent_state("running")

            thinking_cb, flush_thinking = _make_thinking_helpers()
            _tl.thinking_cb = thinking_cb

            def _token_cb(tok: str) -> None:
                _broadcast(("token", tok))

            _tl.token_cb = _token_cb

            def _event_cb(event):
                _push_event(event, None, last_status, flush_thinking=flush_thinking)

            try:
                saved = state_mgr.load(sid)
                ws = WaitState.from_dict(saved.wait_state)
                wake_task = (
                    f"Your scheduled wait has now completed. "
                    f"Original reason for waiting: \"{ws.reason}\". "
                    f"Do NOT output another WAIT \u2014 the timer has expired. "
                    f"Proceed directly with the next step of the task."
                )

                # FIX: use configured permission mode, not hardcoded AUTO
                result = run_agent(
                    task            = wake_task,
                    workspace       = WORKSPACE,
                    permission_mode = _resolve_permission_mode(),
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

                _set_agent_state("waiting" if result.status == "waiting" else "idle")

                label = {
                    "completed":      "done \u2713",
                    "waiting":        f"waiting until {result.wait_until}",
                    "max_iterations": "max iterations reached",
                    "error":          "error",
                    "interrupted":    "interrupted",
                }.get(result.status, result.status)
                flush_thinking()
                _broadcast(("done", label))

            except Exception as exc:
                _broadcast(("error", str(exc)[:200]))
                _broadcast(("done", "error"))
                _set_agent_state("idle")
            finally:
                _tl.token_cb    = None
                _tl.thinking_cb = None
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
    token = request.args.get("token", "")
    if _AGENT_TOKEN and not secrets.compare_digest(token, _AGENT_TOKEN):
        # Build the full auth URL so the QR code works from any device on
        # the network — uses the Host header so it resolves correctly on LAN.
        host = request.host  # e.g. "192.168.1.10:7860"
        proto = "https" if (_SSL_CERT and _SSL_KEY) else "http"
        auth_url = f"{proto}://{host}/?token={_AGENT_TOKEN}"
        unauth = _UNAUTH_HTML.replace("__AUTH_URL__", auth_url)
        return unauth, 401, {"Content-Type": "text/html; charset=utf-8"}
    # Inject the validated token into the page so JS can use it for all
    # subsequent API calls without re-reading it from the URL.
    return HTML.replace("__AGENT_TOKEN__", _AGENT_TOKEN), 200, {
        "Content-Type": "text/html; charset=utf-8"
    }


@_require_auth
@app.route("/stream")
def stream():
    q = _register_stream_q()

    def generate():
        with _agent_state_lock:
            ag_state = _agent_state
        with _session_lock:
            sid = _current_session_id

        connect_payload = {
            "state":   ag_state,
            "session": sid,
            "mode":    _current_permission_mode,
            "history": _chatlog_get(sid),
        }
        yield f"data: {json.dumps({'type': 'connect', 'data': connect_payload}, ensure_ascii=False)}\n\n"

        KA = 15
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


@_require_auth
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

    stop_event = threading.Event()
    with _stop_events_lock:
        _stop_events[rid] = stop_event

    last_status = [""]

    def run_in_thread():
        global _current_session_id
        _set_agent_state("running")

        thinking_cb, flush_thinking = _make_thinking_helpers()
        _tl.thinking_cb = thinking_cb
        _tl.stop_event  = stop_event

        def _token_cb(tok: str) -> None:
            if stop_event.is_set():
                active_resp = getattr(_tl, "current_resp", None)
                if active_resp is not None:
                    try:
                        active_resp.close()
                    except Exception:
                        pass
                raise _AgentStopped("stopped by user")
            _broadcast(("token", tok))

        _tl.token_cb = _token_cb

        def event_cb(event):
            if stop_event.is_set():
                raise _AgentStopped("stopped by user")
            _push_event(event, stop_event, last_status, flush_thinking=flush_thinking)

        _captured_sid: list = [session_id]

        def _whisper_fn() -> "str | None":
            with _whisper_lock:
                return _whisper_store.pop(0) if _whisper_store else None

        def _finalize_session(sid: "str | None") -> None:
            """Update current session ID and merge chat logs atomically."""
            global _current_session_id
            if not sid:
                return
            # FIX: read old_key under the same lock to prevent a race where
            # another thread updates _current_session_id between our read and
            # the merge, causing log entries to be merged into the wrong key.
            with _session_lock:
                old_key = _current_session_id
                _current_session_id = sid
            _chatlog_merge_keys(old_key, sid)
            _broadcast(("session", sid))

        try:
            # FIX: use configured permission mode, not hardcoded AUTO
            result = run_agent(
                task            = user_msg,
                workspace       = WORKSPACE,
                permission_mode = _resolve_permission_mode(),
                resume_session  = session_id,
                mode            = "interactive",
                event_callback  = event_cb,
                soul            = soul,
                whisper_fn      = _whisper_fn,
            )
            _captured_sid[0] = result.session_id

            if not stop_event.is_set():
                _finalize_session(result.session_id)
                _set_agent_state("waiting" if result.status == "waiting" else "idle")

                label = {
                    "completed":      "done \u2713",
                    "waiting":        f"waiting until {result.wait_until}",
                    "max_iterations": "max iterations reached",
                    "error":          "error",
                    "interrupted":    "interrupted",
                    "cancelled":      "cancelled",
                }.get(result.status, result.status)
                flush_thinking()
                _broadcast(("done", label))
            else:
                _finalize_session(result.session_id)
                _set_agent_state("idle")
                _broadcast(("done", "stopped"))

        except _AgentStopped:
            _finalize_session(_captured_sid[0])
            _set_agent_state("idle")
            _broadcast(("done", "stopped"))

        except Exception as exc:
            if not stop_event.is_set():
                _broadcast(("error", str(exc)[:200]))
            _set_agent_state("idle")
        finally:
            _tl.token_cb     = None
            _tl.thinking_cb  = None
            _tl.stop_event   = None
            _tl.current_resp = None
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


@_require_auth
@app.route("/whisper", methods=["POST"])
def whisper():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    with _whisper_lock:
        _whisper_store.append(text)
    return jsonify({"ok": True})


@_require_auth
@app.route("/stop", methods=["POST"])
def stop():
    data = request.get_json(force=True, silent=True) or {}
    rid  = data.get("request_id") or ""
    with _stop_events_lock:
        ev = _stop_events.get(rid)
    if ev:
        ev.set()
    return jsonify({"ok": True})


@_require_auth
@app.route("/new", methods=["POST"])
def new_session():
    global _current_session_id
    with _session_lock:
        _current_session_id = None
    return jsonify({"ok": True})


@_require_auth
@app.route("/sessions")
def sessions():
    session_mgr = SessionManager(WORKSPACE)
    try:
        recent = session_mgr.list_recent(50)
    except Exception:
        recent = []
    return jsonify(recent)


@_require_auth
@app.route("/status")
def status():
    with _session_lock:
        sid = _current_session_id
    return jsonify({
        "workspace":       str(WORKSPACE),
        "llm_url":         getattr(Config, "LLM_URL", ""),
        "current_session": sid,
        "permission_mode": _current_permission_mode,
        "mcp_clients":     len(_global_mcp.clients),
        "agent_state":     _get_agent_state(),
    })


@_require_auth
@app.route("/tools")
def tools():
    return jsonify(_build_tools_payload())


@_require_auth
@app.route("/soul")
def soul_route():
    try:
        text = soul.render() if soul else ""
    except Exception:
        text = ""
    return jsonify({"soul": text})


@_require_auth
@app.route("/mode", methods=["POST"])
def mode():
    global _current_permission_mode
    data = request.get_json(force=True, silent=True) or {}
    m    = (data.get("mode") or "").strip().lower()
    if m not in ("auto", "normal", "manual"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400
    _current_permission_mode = m
    return jsonify({"ok": True, "mode": m})


@_require_auth
@app.route("/cmd", methods=["POST"])
def cmd():
    data = request.get_json(force=True, silent=True) or {}
    c    = (data.get("cmd") or "").strip().lower()
    sid  = data.get("session_id") or None

    if not sid:
        return jsonify({"text": "No session ID provided."})

    if c == "todo":
        try:
            tm    = TodoManager(WORKSPACE, sid)
            items = tm.list()
            if not items:
                return jsonify({"text": "No todos."})
            lines = []
            for t in items:
                mark = "\u2713" if t.get("done") else "\u25a1"
                lines.append(f"  {mark} {t.get('text','')}")
            return jsonify({"text": "\n".join(lines)})
        except Exception as e:
            return jsonify({"text": f"Error: {e}"})

    if c == "plan":
        try:
            pm    = PlanManager(WORKSPACE, sid)
            steps = pm.list()
            if not steps:
                return jsonify({"text": "No active plan."})
            lines = []
            for i, s in enumerate(steps, 1):
                done = s.get("done", False)
                mark = "\u2713" if done else f"{i}."
                lines.append(f"  {mark} {s.get('text','')}")
            return jsonify({"text": "\n".join(lines)})
        except Exception as e:
            return jsonify({"text": f"Error: {e}"})

    return jsonify({"text": f"Unknown command: {c}"})


# =============================================================================
# FILE TREE ROUTES
# =============================================================================

# FIX: include glob patterns alongside plain names so fnmatch can match them.
_HIDDEN_NAMES = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", ".venv",
    "venv", ".env", ".DS_Store", "dist", "build", ".idea", ".vscode",
})
_HIDDEN_GLOBS = ("*.pyc", "*.pyo")   # FIX: use fnmatch, not set membership

_MAX_PREVIEW_BYTES = 100_000


def _safe_rel(path_str: str) -> "Path | None":
    """Resolve a relative path inside WORKSPACE; return None if it escapes."""
    try:
        rel  = Path(path_str)
        full = (WORKSPACE / rel).resolve()
        full.relative_to(WORKSPACE)
        return full
    except (ValueError, Exception):
        return None


def _is_hidden(name: str) -> bool:
    """Return True if a filename should be excluded from the tree."""
    if name in _HIDDEN_NAMES:
        return True
    # FIX: was `name in _HIDDEN` which never matched glob strings like "*.pyc"
    return any(fnmatch.fnmatch(name, pat) for pat in _HIDDEN_GLOBS)


def _ext_of(p: Path) -> str:
    return p.suffix.lower() if p.suffix else ""


@_require_auth
@app.route("/filetree")
def filetree():
    rel  = request.args.get("path", ".").strip() or "."
    full = _safe_rel(rel)
    if full is None:
        return jsonify({"error": "invalid path"}), 400
    if not full.is_dir():
        return jsonify({"error": "not a directory"}), 400

    try:
        items = sorted(full.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return jsonify({"error": "permission denied"}), 403

    entries = []
    for item in items:
        if _is_hidden(item.name) or item.is_symlink():
            continue
        try:
            rel_path = str(item.relative_to(WORKSPACE))
        except ValueError:
            continue

        entry: dict = {
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "path": rel_path,
            "ext":  _ext_of(item) if item.is_file() else "",
        }
        if item.is_file():
            try:
                entry["size"] = item.stat().st_size
            except OSError:
                entry["size"] = 0
        entries.append(entry)

    return jsonify({"entries": entries})


@_require_auth
@app.route("/fileread")
def fileread():
    rel  = request.args.get("path", "").strip()
    if not rel:
        return jsonify({"error": "no path"}), 400
    full = _safe_rel(rel)
    if full is None:
        return jsonify({"error": "invalid path"}), 400
    if not full.is_file():
        return jsonify({"error": "not a file"}), 400

    try:
        size = full.stat().st_size
    except OSError:
        return jsonify({"error": "cannot stat file"}), 500

    try:
        with open(full, "rb") as fh:
            header = fh.read(8192)
    except OSError as e:
        return jsonify({"error": str(e)}), 500

    if b"\x00" in header:
        return jsonify({"binary": True, "size": size})

    truncated = size > _MAX_PREVIEW_BYTES
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(_MAX_PREVIEW_BYTES)
    except OSError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "content":   content,
        "size":      size,
        "truncated": truncated,
        "binary":    False,
    })


@_require_auth
@app.route("/workspace/mtime")
def workspace_mtime():
    """Return the max mtime across the workspace root and its immediate children.

    Used by the FileTree auto-refresh to skip no-op fetches when nothing has
    changed.  One level deep is cheap and catches the vast majority of edits.
    """
    try:
        mtimes = [WORKSPACE.stat().st_mtime]
        for child in WORKSPACE.iterdir():
            try:
                mtimes.append(child.stat().st_mtime)
            except OSError:
                pass
        return jsonify({"mtime": max(mtimes)})
    except OSError:
        return jsonify({"mtime": 0})


# =============================================================================
# STARTUP
# =============================================================================

def _find_free_port(start: int = 7860) -> int:
    # FIX: SO_REUSEADDR prevents false negatives from TIME_WAIT sockets.
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start


if __name__ == "__main__":
    port = _find_free_port(7860)

    sched = threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler")
    sched.start()

    proto = "https" if (_SSL_CERT and _SSL_KEY) else "http"
    print(f"\n  LMAgent Web  v6.6.1")
    print(f"  Workspace : {WORKSPACE}")
    print(f"  Listening : {proto}://{_HOST}:{port}")
    print(f"  Token     : {_AGENT_TOKEN}")
    print(f"  Open      : {proto}://{'localhost' if _HOST == '127.0.0.1' else _HOST}:{port}/?token={_AGENT_TOKEN}\n")

    ssl_context = (_SSL_CERT, _SSL_KEY) if (_SSL_CERT and _SSL_KEY) else None

    app.run(
        host=_HOST,
        port=port,
        threaded=True,
        use_reloader=False,
        debug=False,
        ssl_context=ssl_context,
    )
