"""
LMAgent Web — v7
Place this file next to agent_core.py, agent_tools.py, agent_main.py.
"""
from pathlib import Path as _Path

# ── Load .env file ────────────────────────────────────────────────────────────
def _load_env():
    _dir = _Path(__file__).parent
    env_path = next((p for p in [_dir / ".env", _dir / "env"] if p.exists()), None)
    if not env_path:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in __import__("os").environ:
                    __import__("os").environ[key] = value
    except Exception as e:
        print(f"[web] Warning: could not read env file: {e}")

_load_env()

import functools
import io
import json
import mimetypes as _mimetypes
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

from flask import Flask, Response, jsonify, request

try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ── Core agent imports ────────────────────────────────────────────────────────
try:
    import agent_core
    import agent_tools as _at_mod
    import agent_bca   as _bca_mod
    import agent_main  as _agent_main_mod
    from agent_core import (
        Config, Colors, colored, strip_thinking,
        MCPManager, PermissionMode,
        PlanManager, Safety, SessionManager, StateManager, WaitState,
        SoulConfig, TodoManager, Log, AgentResult,
    )
    from agent_tools import (
        TOOL_SCHEMAS, LLMClient, _REQUIRED_ARG_TOOLS,
        _HeaderStreamCb as _OrigHSC,
    )
    from agent_main import run_agent, _lock_workspace
except ImportError as _ie:
    sys.exit(f"ERROR: agent modules not found.\nDetail: {_ie}")
except Exception:
    sys.exit(f"ERROR importing agent modules:\n{traceback.format_exc()}")

try:
    Config.init()
except Exception as _e:
    sys.exit(f"ERROR: Config.init() failed: {_e}")

WORKSPACE = _lock_workspace(Config.WORKSPACE)
soul      = SoulConfig.load(WORKSPACE)
Log.set_silent(True)

# ── MIME types ────────────────────────────────────────────────────────────────
_mimetypes.add_type("text/javascript",           ".js")
_mimetypes.add_type("text/javascript",           ".mjs")
_mimetypes.add_type("application/json",          ".json")
_mimetypes.add_type("text/css",                  ".css")
_mimetypes.add_type("image/svg+xml",             ".svg")
_mimetypes.add_type("application/manifest+json", ".webmanifest")

_UPLOAD_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

# ── Port ──────────────────────────────────────────────────────────────────────
try:
    _PORT = int(os.environ.get("AGENT_PORT", "7860"))
except ValueError:
    _PORT = 7860

# ── Global MCP (used by tools panel) ─────────────────────────────────────────
_global_mcp = MCPManager(WORKSPACE)
try:
    _global_mcp.load_servers()
except Exception:
    pass

# ── UI template ───────────────────────────────────────────────────────────────
_UI_HTML_PATH = Path(__file__).parent / "agent_web_ui.html"
try:
    HTML = _UI_HTML_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    sys.exit(f"ERROR: UI template not found: {_UI_HTML_PATH}")

# ── Pre-compiled serve rewrite regexes ───────────────────────────────────────
_SRC_HREF_RE = re.compile(r'((?:src|href)\s*=\s*["\'])([^"\']+)(["\'])')
_CSS_URL_RE  = re.compile(r'url\(([^)]+)\)')


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)


@app.after_request
def _security_headers(response):
    h = response.headers
    h["X-Content-Type-Options"] = "nosniff"
    h["Referrer-Policy"]        = "no-referrer"
    if request.path.startswith("/serve/"):
        h["Cross-Origin-Embedder-Policy"] = "credentialless"
        h["Cross-Origin-Opener-Policy"]   = "same-origin"
    else:
        h["X-Frame-Options"]         = "DENY"
        h["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'"
        )
    return response


# =============================================================================
# SECURITY
# =============================================================================

_AGENT_TOKEN = os.environ.get("AGENT_TOKEN") or str(secrets.randbelow(900_000) + 100_000)
print(f"[LMAgent] PIN: {_AGENT_TOKEN}", flush=True)

_HOST     = os.environ.get("AGENT_HOST", "0.0.0.0")
_SSL_CERT = os.environ.get("AGENT_CERT", "")
_SSL_KEY  = os.environ.get("AGENT_KEY",  "")

_rate_data: dict = {}
_rate_lock       = threading.Lock()
_RATE_LIMIT      = 120
_RATE_WINDOW     = 60.0


def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        ts = [t for t in _rate_data.get(ip, []) if now - t < _RATE_WINDOW]
        if len(ts) >= _RATE_LIMIT:
            _rate_data[ip] = ts
            return True
        ts.append(now)
        _rate_data[ip] = ts
        if len(_rate_data) > 1000:
            cutoff = now - _RATE_WINDOW * 2
            for k in [k for k, v in _rate_data.items() if not v or v[-1] < cutoff]:
                del _rate_data[k]
        return False


def _require_auth(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if _AGENT_TOKEN:
            token = (request.headers.get("X-Token", "")
                     or request.args.get("token", ""))
            if not token or not secrets.compare_digest(token, _AGENT_TOKEN):
                return jsonify({"error": "unauthorized"}), 401
        if _is_rate_limited(request.remote_addr or "unknown"):
            return jsonify({"error": "rate limited — try again shortly"}), 429
        return f(*args, **kwargs)
    return wrapper


# =============================================================================
# SIGN-IN PAGE
# =============================================================================

def _make_qr_data_uri(url: str) -> str:
    try:
        import qrcode, qrcode.image.svg, base64
        buf = io.BytesIO()
        qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage,
                    box_size=4, border=2).save(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/svg+xml;base64,{b64}"
    except ImportError:
        return "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'/>"


_UNAUTH_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LMAgent \u2014 Sign In</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0f;color:#8e8c88;font-family:monospace;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}
.box{text-align:center;padding:36px 32px;max-width:360px;width:100%;background:#16161a;border:1px solid #252530;border-radius:12px}
.mark{width:40px;height:40px;background:#e8a245;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:20px;color:#0d0d0f;font-weight:700;margin:0 auto 16px}
h2{color:#ddd8d0;margin-bottom:6px;font-size:15px;font-weight:600}
.sub{font-size:11px;color:#4e4c50;margin-bottom:22px}
.qr-wrap{display:flex;justify-content:center;margin-bottom:20px}
.qr-wrap img{border-radius:8px;border:6px solid #fff;width:180px;height:180px}
.divider{display:flex;align-items:center;gap:10px;margin:18px 0;font-size:10px;color:#4e4c50}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:#252530}
.pin-row{display:flex;gap:8px;justify-content:center;margin-bottom:14px}
.pin-digit{width:44px;height:54px;border-radius:8px;border:1px solid #252530;background:#0d0d0f;
  color:#ddd8d0;font-family:monospace;font-size:22px;font-weight:600;text-align:center;
  outline:none;caret-color:#e8a245;transition:border-color .12s;-webkit-appearance:none}
.pin-digit:focus{border-color:#e8a245}
.pin-digit.filled{border-color:#32323e;color:#e8a245}
.go-btn{width:100%;padding:11px;border-radius:8px;background:#e8a245;border:none;
  color:#0d0d0f;font-family:monospace;font-size:13px;font-weight:600;
  cursor:pointer;transition:background .12s;margin-bottom:10px}
.go-btn:hover{background:#f0b055}
.go-btn:disabled{background:#252530;color:#4e4c50;cursor:default}
.err{font-size:11px;color:#e05555;min-height:16px;margin-top:2px}
.hint{font-size:10px;color:#4e4c50;margin-top:16px;line-height:1.6}
code{background:#0d0d0f;border:1px solid #252530;padding:1px 6px;border-radius:3px;color:#3ecfb2;font-size:10px}
</style>
</head>
<body>
<div class="box">
  <div class="mark">&#955;</div>
  <h2>LMAgent</h2>
  <p class="sub">Scan to open, then enter your PIN</p>
  <div class="qr-wrap"><img src="__QR_IMG__" alt="Scan to open sign-in page"></div>
  <div class="divider">enter PIN</div>
  <div class="pin-row">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
    <input class="pin-digit" maxlength="1" inputmode="numeric" pattern="[0-9]" autocomplete="off">
  </div>
  <button class="go-btn" id="go-btn" disabled onclick="submitPin()">Unlock</button>
  <div class="err" id="err"></div>
  <div class="hint">PIN printed to server console on startup.<br>Set <code>AGENT_TOKEN</code> env var for a fixed PIN.</div>
</div>
<script>
(function(){
  var digits=Array.from(document.querySelectorAll('.pin-digit'));
  var btn=document.getElementById('go-btn');
  function val(){ return digits.map(function(d){return d.value;}).join(''); }
  function upd(){ btn.disabled=val().length!==6; digits.forEach(function(d){d.classList.toggle('filled',d.value!=='');}); }
  digits.forEach(function(d,i){
    d.addEventListener('input',function(){
      d.value=d.value.replace(/\\D/g,'').slice(0,1); upd();
      if(d.value&&i<digits.length-1) digits[i+1].focus();
      if(val().length===6) submitPin();
    });
    d.addEventListener('keydown',function(e){
      if(e.key==='Backspace'&&!d.value&&i>0){digits[i-1].value='';digits[i-1].focus();upd();}
    });
    d.addEventListener('paste',function(e){
      e.preventDefault();
      var t=(e.clipboardData||window.clipboardData).getData('text').replace(/\\D/g,'').slice(0,6);
      t.split('').forEach(function(ch,j){if(digits[i+j])digits[i+j].value=ch;});
      upd(); digits[Math.min(i+t.length,digits.length-1)].focus();
      if(val().length===6) submitPin();
    });
  });
  digits[0].focus();
  window.submitPin=function(){
    var p=val(); if(p.length!==6)return;
    btn.disabled=true;
    window.location.href=window.location.origin+window.location.pathname+'?token='+encodeURIComponent(p);
  };
})();
</script>
</body>
</html>"""


# =============================================================================
# GLOBAL STATE
# =============================================================================

_AGENT_LOCK         = threading.Lock()
_AGENT_LOCK_TIMEOUT = 30
_tl                 = threading.local()

_active_responses:      dict           = {}
_active_responses_lock: threading.Lock = threading.Lock()

_session_lock            = threading.Lock()
_current_session_id      = None
_current_permission_mode = Config.PERMISSION_MODE

_stop_events:      dict           = {}
_stop_events_lock: threading.Lock = threading.Lock()

_whisper_store: list = []
_whisper_lock        = threading.Lock()

_agent_state      = "idle"
_agent_state_lock = threading.Lock()

# Messaging mode — "web" means only the browser UI can send tasks;
# any other value ("discord", "telegram", "whatsapp", "sms") means
# the named platform is the active input source.  agent_messaging.py
# calls _set_messaging_mode() when the operator switches platforms via
# the /messaging/mode API endpoint.
_messaging_mode      = "web"
_messaging_mode_lock = threading.Lock()

_PERM_MODE_MAP = {m.value: m for m in PermissionMode}


def _set_agent_state(s: str) -> None:
    global _agent_state
    with _agent_state_lock:
        _agent_state = s


def _get_agent_state() -> str:
    with _agent_state_lock:
        return _agent_state


def _get_messaging_mode() -> str:
    with _messaging_mode_lock:
        return _messaging_mode


def _set_messaging_mode(mode: str) -> None:
    global _messaging_mode
    with _messaging_mode_lock:
        _messaging_mode = mode


def _resolve_permission_mode() -> PermissionMode:
    return _PERM_MODE_MAP.get(
        (_current_permission_mode or "").lower(), PermissionMode.AUTO
    )


# =============================================================================
# SSE BROADCAST / STREAM QUEUES
# =============================================================================

_stream_queues:      list           = []
_stream_queues_lock: threading.Lock = threading.Lock()


def _broadcast(item: tuple) -> None:
    """Push an event to all connected SSE clients and the persistent chat log."""
    # Fix: normalize any latin-1 / utf-8 encoding edge cases (from script v2)
    kind, payload = item
    if isinstance(payload, str):
        try:
            payload = payload.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        item = (kind, payload)
    _chatlog_append(item)
    with _stream_queues_lock:
        queues = list(_stream_queues)
    for q in queues:
        try:
            q.put_nowait(item)
        except Exception:
            pass


def _register_stream_q() -> queue.Queue:
    q = queue.Queue()
    with _stream_queues_lock:
        _stream_queues.append(q)
    return q


def _unregister_stream_q(q: queue.Queue) -> None:
    with _stream_queues_lock:
        try:
            _stream_queues.remove(q)
        except ValueError:
            pass


# =============================================================================
# PERSISTENT CHAT LOG
# =============================================================================

_chat_logs:     dict           = defaultdict(list)
_chat_log_lock: threading.Lock = threading.Lock()
_NO_SESSION_KEY = "_none_"
_REPLAY_KINDS   = frozenset({
    "token", "thinking", "tool", "status",
    "iteration", "done", "error", "session",
})


def _clog_key(sid) -> str:
    return sid if sid else _NO_SESSION_KEY


def _chatlog_append(item: tuple) -> None:
    if item[0] not in _REPLAY_KINDS:
        return
    key = _clog_key(_current_session_id)
    with _chat_log_lock:
        _chat_logs[key].append(item)


def _chatlog_get(sid) -> list:
    with _chat_log_lock:
        return list(_chat_logs.get(_clog_key(sid), []))


def _chatlog_clear(sid) -> None:
    with _chat_log_lock:
        _chat_logs.pop(_clog_key(sid), None)


def _chatlog_merge_keys(old_sid, new_sid) -> None:
    ok, nk = _clog_key(old_sid), _clog_key(new_sid)
    if ok == nk:
        return
    with _chat_log_lock:
        if ok in _chat_logs:
            _chat_logs[nk] = _chat_logs.pop(ok, []) + _chat_logs.get(nk, [])


# =============================================================================
# PATH VALIDATION HELPER
# =============================================================================

def _validate_ft_path(path: str):
    """Validate a workspace-relative path; returns (ok, err, resolved_path)."""
    if not path or path in (".", ""):
        return True, "", WORKSPACE
    ok, err, fp = Safety.validate_path(WORKSPACE, path)
    if not ok:
        return False, err, WORKSPACE
    return True, "", fp


# =============================================================================
# BCA WEB HOOKS
# =============================================================================

_orig_run_bca_agent     = _bca_mod._run_bca_agent
_orig_dispatch_bca_tool = _bca_mod._dispatch_bca_tool
_orig_tool_decompose    = _bca_mod.tool_decompose
_orig_tool_delegate     = _bca_mod.tool_delegate

_BCA_SILENT_TOOLS = frozenset({
    "task_state_update", "task_state_get", "task_reconcile", "todo_list",
})


def _web_run_bca_agent(brief, workspace, bm, parent_ctx,
                       stream_callback=None, max_iterations=None):
    depth  = getattr(brief, "depth", 0)
    indent = "  " * max(0, depth - 1)
    short  = brief.agent_id[:8]
    obj    = brief.objective[:60]

    _broadcast(("status", f"{indent}▶ [{short}] {obj}…"))

    prev_token_cb = getattr(_tl, "token_cb", None)
    _tl.token_cb  = None

    try:
        result = _orig_run_bca_agent(
            brief, workspace, bm, parent_ctx,
            stream_callback=None,
            max_iterations=max_iterations,
        )
    finally:
        _tl.token_cb = prev_token_cb

    status  = result.get("status", "error")
    icon    = {"ok": "✓", "partial": "◑"}.get(status, "✗")
    summary = (result.get("summary") or "")[:60]
    arts    = result.get("artifacts") or []
    line    = f"{indent}{icon} [{short}] {summary}"
    if arts:
        line += f" → {', '.join(arts[:3])}"
    _broadcast(("status", line))
    return result


def _web_dispatch_bca_tool(fn_name, args, workspace, brief=None):
    if fn_name not in _BCA_SILENT_TOOLS:
        preview = ""
        for key in ("path", "command", "pattern", "step_id", "description"):
            v = args.get(key, "")
            if v:
                preview = str(v)[:50]
                break
        _broadcast(("tool", f"{fn_name}({preview})"))

    result = _orig_dispatch_bca_tool(fn_name, args, workspace, brief=brief)

    if fn_name not in _BCA_SILENT_TOOLS:
        ok = result.get("success", False)
        _broadcast(("tool", f"{'✓' if ok else '✗'} {fn_name}"))

    return result


def _web_tool_decompose(workspace, manifest_json):
    try:
        if isinstance(manifest_json, dict):
            n = len(manifest_json.get("tasks", []))
        elif isinstance(manifest_json, list):
            n = len(manifest_json)
        else:
            raw = str(manifest_json).strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw   = "\n".join(lines[1:])
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3].rstrip()
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            n   = len(json.loads(raw).get("tasks", []))
        _broadcast(("status", f"decomposing into {n} sub-task(s)…"))
    except Exception:
        _broadcast(("status", "decomposing task…"))

    result = _orig_tool_decompose(workspace, manifest_json)

    ok  = result.get("tasks_ok",  0)
    ran = result.get("tasks_run", 0)
    if result.get("success"):
        _broadcast(("status", f"✓ decompose: all {ok} task(s) succeeded"))
    elif ok > 0:
        _broadcast(("status", f"◑ decompose: {ok}/{ran} task(s) succeeded"))
    else:
        _broadcast(("status", f"✗ decompose failed: {result.get('error','')[:80]}"))

    return result


def _web_tool_delegate(workspace, **kwargs):
    obj = kwargs.get("objective", "")[:60]
    _broadcast(("status", f"delegate → {obj}…"))

    result = _orig_tool_delegate(workspace, **kwargs)

    ok      = result.get("success", False)
    summary = (result.get("summary") or "")[:60]
    arts    = result.get("artifacts") or []
    line    = f"{'✓' if ok else '✗'} delegate: {summary}"
    if arts:
        line += f" → {', '.join(arts[:3])}"
    _broadcast(("status", line))
    return result


# Apply patches
_bca_mod._run_bca_agent     = _web_run_bca_agent
_bca_mod._dispatch_bca_tool = _web_dispatch_bca_tool
_bca_mod.tool_decompose      = _web_tool_decompose
_bca_mod.tool_delegate       = _web_tool_delegate

_at_mod.TOOL_HANDLERS["decompose"] = _web_tool_decompose
_at_mod.TOOL_HANDLERS["delegate"]  = _web_tool_delegate


# =============================================================================
# LLMClient STREAM PARSER PATCH
# =============================================================================

def _web_parse_stream(resp, stream_callback):
    content       = ""
    tool_calls    = {}
    next_idx      = 0
    finish_reason = None
    in_think      = False

    resp.encoding = "utf-8"
    req_id = getattr(_tl, "request_id", None)
    if req_id:
        with _active_responses_lock:
            _active_responses[req_id] = resp
    _tl.current_resp = resp

    stop_event = getattr(_tl, "stop_event", None)

    try:
        for line in resp.iter_lines(decode_unicode=True):
            if stop_event and stop_event.is_set():
                try:
                    resp.close()
                except Exception:
                    pass
                break

            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
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
                content    += raw_content
                eff_cb      = stream_callback or getattr(_tl, "token_cb", None)
                thinking_cb = getattr(_tl, "thinking_cb", None)

                if eff_cb or thinking_cb:
                    for part in re.split(r"(</?think>)", raw_content, flags=re.IGNORECASE):
                        if not part:
                            continue
                        if part.lower() == "<think>":
                            in_think = True
                        elif part.lower() == "</think>":
                            in_think = False
                            if thinking_cb:
                                thinking_cb(None)
                        elif in_think:
                            sys.stdout.write(colored(part, Colors.GRAY))
                            sys.stdout.flush()
                            if thinking_cb:
                                thinking_cb(part)
                        elif eff_cb:
                            eff_cb(part)

            for tb in (delta.get("thinking") or []):
                tb_text = tb.get("thinking") if isinstance(tb, dict) else None
                if tb_text:
                    thinking_cb = getattr(_tl, "thinking_cb", None)
                    sys.stdout.write(colored(tb_text, Colors.GRAY))
                    sys.stdout.flush()
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

    except (AttributeError, ConnectionError, OSError) as _stream_err:
        Log.warning(
            f"[web] Stream interrupted mid-response "
            f"({type(_stream_err).__name__}: {_stream_err}) — returning partial result"
        )
        _broadcast(("status", "⚠ stream interrupted — partial response returned"))

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
        is_empty = not args_str or not args_str.strip()

        if is_empty or args_str.strip() == "{}":
            if fn_name in _REQUIRED_ARG_TOOLS:
                Log.error(f"'{fn_name}' received empty args — finish_reason={finish_reason!r}")
                incomplete                  = True
                tc["function"]["arguments"] = "{}"
                tc["_truncated"]            = True
            else:
                tc["function"]["arguments"] = "{}"
            calls.append(tc)
            continue

        try:
            json.loads(args_str)
            calls.append(tc)
        except json.JSONDecodeError as parse_err:
            Log.error(f"[PARSE] '{fn_name}' JSON decode failed: {parse_err.msg}")
            repaired = False
            if len(args_str) < 500:
                opens, closes = args_str.count("{"), args_str.count("}")
                if opens > closes:
                    candidate = args_str + "}" * (opens - closes)
                    try:
                        json.loads(candidate)
                        tc["function"]["arguments"] = candidate
                        incomplete = True
                        calls.append(tc)
                        repaired = True
                    except json.JSONDecodeError:
                        pass
            if not repaired:
                incomplete                  = True
                tc["function"]["arguments"] = "{}"
                tc["_truncated"]            = True
                calls.append(tc)

    if finish_reason == "length":
        Log.warning("⚠️  Generation stopped: output token limit hit")
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


LLMClient._parse_stream = staticmethod(_web_parse_stream)


# =============================================================================
# STREAMING HEADER CALLBACK PATCH
# =============================================================================

class _PatchedHSC(_OrigHSC):
    def __call__(self, token: str) -> None:
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


_agent_main_mod._HeaderStreamCb = _PatchedHSC
_at_mod._HeaderStreamCb         = _PatchedHSC


# =============================================================================
# STOP SIGNAL
# =============================================================================

class _AgentStopped(BaseException):
    """Raised inside token_cb to abort a running agent mid-stream."""


# =============================================================================
# THINKING HELPERS
# =============================================================================

def _make_thinking_helpers():
    buf = [""]

    def flush_thinking() -> None:
        text = buf[0].strip()
        if text:
            _broadcast(("thinking", text))
        buf[0] = ""

    def thinking_cb(part) -> None:
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


def _push_event(event, stop_event, last_status: list, flush_thinking=None) -> None:
    if stop_event and stop_event.is_set():
        return
    etype, edata = event.type, event.data

    if etype == "tool_call":
        if flush_thinking:
            flush_thinking()
        _broadcast(("tool",
                    f"{edata.get('name','?')}({edata.get('args_preview','')[:50]})"))

    elif etype == "tool_result":
        _broadcast(("tool",
                    f"{'✓' if edata.get('success') else '✗'} {edata.get('name','')}"))

    elif etype == "iteration":
        if flush_thinking:
            flush_thinking()
        _broadcast(("iteration", f"{edata.get('n')}/{edata.get('max')}"))

    elif etype in ("log", "warning", "error"):
        msg = edata.get("message") or edata.get("error") or ""
        if msg and (etype == "error" or (not _is_noisy(msg) and msg != last_status[0])):
            last_status[0] = msg
            _broadcast(("status", msg[:120]))

    elif etype == "waiting":
        _broadcast(("status", f"waiting until {edata.get('resume_after')}"))

    elif etype == "complete":
        _broadcast(("status", f"done — {edata.get('reason', '')}"))


def _sse_response(generator):
    return Response(generator, mimetype="text/event-stream", headers={
        "Cache-Control":     "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection":        "keep-alive",
        "Content-Type":      "text/event-stream; charset=utf-8",
    })


# =============================================================================
# SHARED AGENT EXECUTION
# =============================================================================

# PATCH 1 & 2 & 3: Added web_origin parameter; gated session broadcasts and
# _current_session_id updates so only web-originated calls affect browser state.
def _execute_agent(message, session_id, request_id, stop_ev,
                   permission_mode, whisper_fn=None,
                   web_origin: bool = True, on_session=None):
    global _current_session_id

    _tl.stop_event  = stop_ev
    _tl.request_id  = request_id

    thinking_cb, flush_thinking = _make_thinking_helpers()
    _tl.thinking_cb = thinking_cb

    def token_cb(tok: str) -> None:
        if stop_ev.is_set():
            raise _AgentStopped()
        _broadcast(("token", tok))

    _tl.token_cb = token_cb

    # PATCH 2: Only broadcast the initial "session" event for web-originated calls.
    # Messaging/scheduler tasks must not push their session IDs to SSE clients.
    if web_origin:
        _broadcast(("session", session_id or ""))
    _set_agent_state("running")
    last_status = [""]

    def ev_cb(event):
        if stop_ev.is_set():
            raise _AgentStopped()   # fires between tool calls, not just during streaming
        _push_event(event, stop_ev, last_status, flush_thinking)

    def whisper_fn_safe():
        with _whisper_lock:
            return _whisper_store.pop(0) if _whisper_store else None

    try:
        result = run_agent(
            message, WORKSPACE,
            permission_mode=permission_mode,
            resume_session=session_id,
            plan_first=False,
            mode="output",
            event_callback=ev_cb,
            soul=soul,
            whisper_fn=whisper_fn or whisper_fn_safe,
        )
        flush_thinking()

        # PATCH 3: Only update _current_session_id and broadcast the final
        # session event when the call originated from the web /chat endpoint.
        with _session_lock:
            if web_origin:
                _current_session_id = result.session_id
        if web_origin:
            _chatlog_merge_keys(session_id, result.session_id)
            _broadcast(("session", result.session_id))
        # Always notify non-web callers of the final session ID
        if on_session and result.session_id:
            on_session(result.session_id)

        if result.status == "waiting":
            _broadcast(("done", f"waiting until {result.wait_until}"))
            _set_agent_state("waiting")
        elif result.status in ("error", "interrupted"):
            _broadcast(("error", result.final_answer or result.status))
            _set_agent_state("idle")
        else:
            words  = (result.final_answer or "").split()
            reason = " ".join(words[:8]) + ("…" if len(words) > 8 else "")
            _broadcast(("done", reason or result.status))
            _set_agent_state("idle")

    except _AgentStopped:
        flush_thinking()
        # Re-confirm the session ID so the next message picks it up cleanly
        if web_origin:
            _broadcast(("session", session_id or ""))
        _broadcast(("done", "stopped"))
        _set_agent_state("idle")

    except Exception as e:
        flush_thinking()
        _broadcast(("error", str(e)))
        _set_agent_state("idle")
        Log.error(f"[agent] unhandled exception: {e}")
        traceback.print_exc()

    finally:
        _tl.stop_event  = None
        _tl.request_id  = None
        _tl.thinking_cb = None
        _tl.token_cb    = None
        with _active_responses_lock:
            _active_responses.pop(request_id, None)


# =============================================================================
# TOOL CATEGORIES
# =============================================================================

_TOOL_CATEGORIES = {
    "Files":      ["read", "write", "edit", "glob", "grep", "ls", "mkdir"],
    "Git":        ["git_status", "git_diff", "git_add", "git_commit", "git_branch"],
    "System":     ["shell", "get_time"],
    "Tasks":      ["todo_add", "todo_complete", "todo_update", "todo_list",
                   "plan_complete_step", "task"],
    "Delegation": ["delegate", "decompose", "report_result"],
    "State":      ["task_state_update", "task_state_get", "task_reconcile"],
}


def _build_tools_payload() -> dict:
    schema_map = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
    builtin    = {}
    for cat, names in _TOOL_CATEGORIES.items():
        tools = []
        for name in names:
            fn = schema_map.get(name)
            if fn:
                tools.append({
                    "name":        name,
                    "description": fn.get("description", ""),
                    "params":      list(fn.get("parameters", {}).get("properties", {}).keys()),
                    "required":    fn.get("parameters", {}).get("required", []),
                })
        if tools:
            builtin[cat] = tools

    mcp_groups = {}
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
            mcp_groups[f"MCP · {client.name}"] = tools

    return {"builtin": builtin, "mcp": mcp_groups}


# =============================================================================
# SERVE FILE — module-level rewrite helpers
# =============================================================================

def _rewrite_html_asset(m, base_dir: str, token: str) -> str:
    attr  = m.group(1)
    path  = m.group(2)
    quote = m.group(3)
    if path.startswith(("http://", "https://", "//", "data:", "javascript:", "#", "/serve/")):
        return m.group(0)
    serve_path = f"/serve{path}" if path.startswith("/") else \
                 f"/serve/{base_dir + '/' if base_dir else ''}{path}"
    sep = "&" if "?" in serve_path else "?"
    return f"{attr}{serve_path}{sep}token={token}{quote}"


def _rewrite_css_asset(m, base_dir: str, token: str) -> str:
    path = m.group(1).strip("'\"")
    if path.startswith(("http://", "https://", "//", "data:", "/serve/")):
        return m.group(0)
    serve_path = f"/serve{path}" if path.startswith("/") else \
                 f"/serve/{base_dir + '/' if base_dir else ''}{path}"
    sep = "&" if "?" in serve_path else "?"
    return f"url('{serve_path}{sep}token={token}')"


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/command", methods=["POST"])
@_require_auth
def slash_command():
    global _current_permission_mode
    data       = request.get_json(force=True, silent=True) or {}
    cmd        = (data.get("command") or "").strip()
    session_id = data.get("session_id") or _current_session_id

    if not cmd.startswith("/"):
        return jsonify({"error": "not a command"}), 400

    parts    = cmd.split(maxsplit=1)
    cmd_name = parts[0].lower()
    cmd_arg  = parts[1].strip() if len(parts) > 1 else ""

    if cmd_name == "/help":
        output = (
            "Available Commands\n"
            "──────────────────────────────────────────────────────\n"
            "  /help           This help screen\n"
            "  /sessions       List recent sessions\n"
            "  /mode <level>   Set permission: manual / normal / auto\n"
            "  /plan           Show the current plan\n"
            "  /todo           Show todo list\n"
            "  /status         Show agent + MCP status\n"
            "  /soul           Show loaded soul / personality\n"
            "  /new            Start a completely fresh session\n"
            "  /session        Show current session ID"
        )

    elif cmd_name == "/sessions":
        sessions = SessionManager(WORKSPACE).list_recent(10)
        if not sessions:
            output = "No sessions found."
        else:
            lines = ["Recent Sessions", "─" * 60]
            for s in sessions:
                parent = f"  ← {s['parent'][:8]}" if s.get("parent") else ""
                iters  = f"  ·  {s['iterations']} iter" if s.get("iterations") else ""
                lines.append(
                    f"  {s['id'][-12:]}  {s['status']:<12}  "
                    f"{s['task'][:38]}{iters}{parent}"
                )
            output = "\n".join(lines)

    elif cmd_name == "/status":
        mcp_lines = ""
        status_map = _global_mcp.get_status()
        if status_map:
            mcp_lines = "\n  MCP Servers:\n" + "\n".join(
                f"    {'✓' if ok else '✗'} {name}"
                for name, ok in status_map.items()
            )
        output = (
            f"Agent Status\n"
            f"────────────────────────────────────────────\n"
            f"  Workspace  : {WORKSPACE}\n"
            f"  LLM        : {Config.LLM_URL}\n"
            f"  Session    : {(session_id or 'none')[-8:]}\n"
            f"  State      : {_get_agent_state()}\n"
            f"  Permissions: {_current_permission_mode}"
            f"{mcp_lines}"
        )

    elif cmd_name == "/soul":
        loaded_soul = SoulConfig.load(WORKSPACE)
        output = (
            f"Soul Config  (.soul.md)\n"
            f"────────────────────────────────────────────\n"
            f"{loaded_soul or '(no .soul.md found in workspace)'}\n\n"
            f"Edit .soul.md in your workspace to customise personality."
        )

    elif cmd_name == "/session":
        if session_id:
            output = f"Current session: {session_id}"
        else:
            output = "No active session — send your first task to create one."

    elif cmd_name == "/plan":
        if not session_id:
            output = "No active session."
        else:
            try:
                pm   = PlanManager(WORKSPACE, session_id)
                plan = pm.plan
                if plan:
                    output = f"Current Plan\n{'─'*40}\n{json.dumps(plan, indent=2)}"
                else:
                    output = "No active plan for this session."
            except Exception as e:
                output = f"Could not load plan: {e}"

    elif cmd_name == "/todo":
        if not session_id:
            output = "No active session."
        else:
            try:
                tm    = TodoManager(WORKSPACE, session_id)
                todos = getattr(tm, "todos", None) or []
                if todos:
                    lines = [f"Todo List  ({len(todos)} items)", "─" * 40]
                    for t in todos:
                        mark = "✓" if t.get("done") or t.get("completed") else "○"
                        lines.append(f"  {mark}  {t.get('text', t.get('task', ''))}")
                    output = "\n".join(lines)
                else:
                    output = "No todos for this session."
            except Exception as e:
                output = f"Could not load todos: {e}"

    elif cmd_name == "/mode":
        if cmd_arg in ("manual", "normal", "auto"):
            _current_permission_mode = cmd_arg
            output = f"✓ Permission mode → {cmd_arg}"
        else:
            output = "Usage: /mode <manual|normal|auto>"

    else:
        output = f"Unknown command: {cmd_name}  —  type /help for the full list."

    return jsonify({"ok": True, "output": output})


@app.route("/")
def index():
    if _AGENT_TOKEN:
        token = (request.headers.get("X-Token", "")
                 or request.args.get("token", ""))
        if not token or not secrets.compare_digest(token, _AGENT_TOKEN):
            proto = "https" if _SSL_CERT and _SSL_KEY else "http"
            try:
                host = request.host or f"localhost:{_PORT}"
            except Exception:
                host = f"localhost:{_PORT}"
            base_url = f"{proto}://{host}/"
            qr_img   = _make_qr_data_uri(base_url)
            return _UNAUTH_HTML.replace("__QR_IMG__", qr_img), 401
    return Response(
        HTML.replace("__AGENT_TOKEN__", _AGENT_TOKEN or ""),
        mimetype="text/html",
    )


@app.route("/chat", methods=["POST"])
@_require_auth
def chat():
    global _current_session_id, _current_permission_mode

    data       = request.get_json(force=True, silent=True) or {}
    message    = (data.get("message") or "").strip()
    session_id = data.get("session_id") or None
    request_id = data.get("request_id") or secrets.token_hex(8)

    if not message:
        return jsonify({"error": "empty message"}), 400

    if not _AGENT_LOCK.acquire(timeout=_AGENT_LOCK_TIMEOUT):
        return jsonify({"error": "agent busy — try again shortly"}), 429

    stop_ev = threading.Event()
    with _stop_events_lock:
        _stop_events[request_id] = stop_ev

    perm_mode = _resolve_permission_mode()

    with _session_lock:
        _current_session_id = session_id

    def _run():
        try:
            _execute_agent(message, session_id, request_id, stop_ev, perm_mode)
        finally:
            _AGENT_LOCK.release()
            with _stop_events_lock:
                _stop_events.pop(request_id, None)

    threading.Thread(
        target=_run, daemon=True,
        name=f"agent-{request_id[:8]}",
    ).start()
    return jsonify({"ok": True, "request_id": request_id})


@app.route("/stop", methods=["POST"])
@_require_auth
def stop():
    data       = request.get_json(force=True, silent=True) or {}
    request_id = data.get("request_id", "")

    with _active_responses_lock:
        resp = _active_responses.get(request_id)
    if resp:
        try:
            resp.close()
        except Exception:
            pass

    with _stop_events_lock:
        ev = _stop_events.get(request_id)
    if ev:
        ev.set()

    # Give the agent thread up to 3 s to release the lock so the next
    # /chat call doesn't immediately bounce with 429.
    acquired = _AGENT_LOCK.acquire(timeout=3)
    if acquired:
        _AGENT_LOCK.release()

    return jsonify({"ok": True})

@app.route("/new", methods=["POST"])
@_require_auth
def new_session():
    global _current_session_id
    with _session_lock:
        _current_session_id = None
    _set_agent_state("idle")
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    token = (request.headers.get("X-Token", "")
             or request.args.get("token", ""))
    if _AGENT_TOKEN and not secrets.compare_digest(token, _AGENT_TOKEN):
        return jsonify({"error": "unauthorized"}), 401

    def _gen():
        q   = _register_stream_q()
        sid = _current_session_id

        history = _chatlog_get(sid)
        connect_payload = {
            "session": sid or "",
            "state":   _get_agent_state(),
            "mode":    _current_permission_mode,
            "history": [[k, v] for k, v in history],
        }
        yield f"data: {json.dumps({'type': 'connect', 'data': connect_payload})}\n\n"

        try:
            while True:
                try:
                    kind, payload = q.get(timeout=20)
                    yield f"data: {json.dumps({'type': kind, 'data': payload})}\n\n"
                except Exception:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            _unregister_stream_q(q)

    return _sse_response(_gen())


@app.route("/status")
@_require_auth
def status():
    with _session_lock:
        sid = _current_session_id
    return jsonify({
        "state":      _get_agent_state(),
        "session_id": sid,
        "workspace":  str(WORKSPACE),
        "mode":       _current_permission_mode,
    })


@app.route("/sessions")
@_require_auth
def sessions():
    mgr  = SessionManager(WORKSPACE)
    sess = mgr.list_recent(20)
    return jsonify({"sessions": sess})


@app.route("/upload", methods=["POST"])
@_require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f    = request.files["file"]
    name = f.filename or "upload"
    ext  = Path(name).suffix.lower()
    if ext not in _UPLOAD_EXTS:
        return jsonify({"error": f"unsupported extension: {ext}"}), 400

    safe_name = re.sub(r"[^\w.\-]", "_", Path(name).stem) + ext
    dest      = WORKSPACE / safe_name
    counter   = 0
    while dest.exists():
        counter += 1
        dest = WORKSPACE / f"{Path(name).stem}_{counter}{ext}"

    if _PIL_AVAILABLE:
        try:
            img = _PIL_Image.open(f.stream)
            img.verify()
            f.stream.seek(0)
            img = _PIL_Image.open(f.stream)
            if img.width > 4096 or img.height > 4096:
                img.thumbnail((2048, 2048), _PIL_Image.LANCZOS)
            img.save(str(dest))
        except Exception as e:
            return jsonify({"error": f"invalid image: {e}"}), 400
    else:
        f.save(str(dest))

    return jsonify({
        "ok":       True,
        "path":     str(dest.relative_to(WORKSPACE)).replace("\\", "/"),
        "filename": dest.name,
    })


@app.route("/whisper", methods=["POST"])
@_require_auth
def whisper():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    with _whisper_lock:
        _whisper_store.append(text)
    return jsonify({"ok": True})


@app.route("/filetree")
@_require_auth
def filetree():
    path = request.args.get("path", ".").strip()
    ok, err, dp = _validate_ft_path(path)
    if not ok:
        return jsonify({"error": err}), 400
    if not dp.is_dir():
        return jsonify({"error": "not a directory"}), 400
    try:
        entries = []
        for item in sorted(dp.iterdir()):
            if item.name.startswith("."):
                continue
            try:
                stat = item.stat()
                rel  = str(item.relative_to(WORKSPACE)).replace("\\", "/")
                entries.append({
                    "name":  item.name,
                    "path":  rel,
                    "type":  "dir" if item.is_dir() else "file",
                    "size":  stat.st_size if item.is_file() else 0,
                    "ext":   item.suffix.lower() if item.is_file() else "",
                    "mtime": stat.st_mtime,
                })
            except Exception:
                continue
        return jsonify({"entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fileread")
@_require_auth
def fileread():
    path = request.args.get("path", "").strip()
    ok, err, fp = _validate_ft_path(path)
    if not ok:
        return jsonify({"error": err}), 400
    if not fp.is_file():
        return jsonify({"error": "not a file"}), 400

    MAX_PREVIEW = 100_000
    if fp.suffix.lower() in Config.BINARY_EXTS:
        return jsonify({"binary": True, "size": fp.stat().st_size})
    try:
        raw  = fp.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return jsonify({
            "content":   text[:MAX_PREVIEW],
            "truncated": len(text) > MAX_PREVIEW,
            "size":      len(raw),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/workspace/mtime")
@_require_auth
def workspace_mtime():
    try:
        mtime = max(
            (p.stat().st_mtime for p in WORKSPACE.rglob("*")
             if p.is_file()
             and not any(part.startswith(".") for part in p.relative_to(WORKSPACE).parts)),
            default=0,
        )
        return jsonify({"mtime": mtime})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/serve/<path:rel_path>")
def serve_file(rel_path: str):
    token = (request.headers.get("X-Token", "")
             or request.args.get("token", "")
             or request.cookies.get("agent_token", ""))
    if _AGENT_TOKEN and not secrets.compare_digest(token, _AGENT_TOKEN):
        return Response("Unauthorized", status=401, mimetype="text/plain")

    ok, err, fp = _validate_ft_path(rel_path)
    if not ok or not fp.is_file():
        return Response("Not found", status=404, mimetype="text/plain")

    mime = _mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
    try:
        raw = fp.read_bytes()

        if mime == "text/html":
            html     = raw.decode("utf-8", errors="replace")
            base_dir = str(fp.parent.relative_to(WORKSPACE)).replace("\\", "/")
            if base_dir == ".":
                base_dir = ""
            html = _SRC_HREF_RE.sub(
                lambda m: _rewrite_html_asset(m, base_dir, token), html
            )
            return Response(html.encode("utf-8"), mimetype="text/html")

        if mime == "text/css":
            css      = raw.decode("utf-8", errors="replace")
            base_dir = str(fp.parent.relative_to(WORKSPACE)).replace("\\", "/")
            if base_dir == ".":
                base_dir = ""
            css = _CSS_URL_RE.sub(
                lambda m: _rewrite_css_asset(m, base_dir, token), css
            )
            return Response(css.encode("utf-8"), mimetype="text/css")

        return Response(raw, mimetype=mime)

    except PermissionError:
        return Response("Permission denied", status=403, mimetype="text/plain")
    except Exception as exc:
        return Response(f"Error: {exc}", status=500, mimetype="text/plain")


@app.route("/tools")
@_require_auth
def tools():
    return jsonify(_build_tools_payload())


# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================

def _start_web_scheduler() -> None:
    def _sched_run_agent(
        task,
        workspace,
        permission_mode=PermissionMode.AUTO,
        resume_session=None,
        plan_first=False,
        mode="interactive",
        event_callback=None,
        soul="",
        whisper_fn=None,
    ):
        request_id = f"sched_{secrets.token_hex(6)}"
        stop_ev    = threading.Event()

        with _stop_events_lock:
            _stop_events[request_id] = stop_ev

        if not _AGENT_LOCK.acquire(timeout=_AGENT_LOCK_TIMEOUT):
            Log.warning("[web-scheduler] agent busy — skipping scheduled task")
            return AgentResult(
                status="error",
                final_answer="agent busy",
                events=[],
                session_id=resume_session or "",
                iterations=0,
            )

        try:
            # PATCH 4: Pass web_origin=False so the scheduler cannot clobber
            # the browser's session state.
            _execute_agent(task, resume_session, request_id, stop_ev,
                           permission_mode, web_origin=False)
            with _session_lock:
                sid = _current_session_id or resume_session or ""
            status = "waiting" if _get_agent_state() == "waiting" else "completed"
            return AgentResult(
                status=status,
                final_answer="",
                events=[],
                session_id=sid,
                iterations=0,
            )
        finally:
            _AGENT_LOCK.release()
            with _stop_events_lock:
                _stop_events.pop(request_id, None)

    _agent_main_mod.run_agent = _sched_run_agent

    def _run():
        # Silence verbose internal logs on the scheduler thread
        Log.set_silent(True)
        try:
            _agent_main_mod.run_scheduler(WORKSPACE, soul=soul)
        except Exception as e:
            Log.error(f"[web scheduler] crashed: {e}")

    threading.Thread(target=_run, daemon=True, name="web-scheduler").start()
    Log.info("[web] Background scheduler started")


# =============================================================================
# MESSAGING INTEGRATION
# =============================================================================

try:
    from agent_messaging import init_messaging as _init_messaging
    _MESSAGING_AVAILABLE = True
except ImportError:
    _MESSAGING_AVAILABLE = False
    print("[messaging] agent_messaging.py not found — messaging disabled")

_MESSAGING_CFG = None   # populated inside main()


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main() -> None:
    global _MESSAGING_CFG

    Log.set_silent(False)

    proto = "https" if _SSL_CERT and _SSL_KEY else "http"
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "localhost"

    print(f"\n{'=' * 60}")
    print(f"  LMAgent Web Interface v7.0")
    print(f"{'=' * 60}")
    print(f"  Local  : {proto}://localhost:{_PORT}")
    print(f"  Network: {proto}://{local_ip}:{_PORT}")
    print(f"  PIN    : {_AGENT_TOKEN}")
    print(f"  WS     : {WORKSPACE}")
    print(f"{'=' * 60}\n")

    _start_web_scheduler()

    # ── Start messaging integrations ──────────────────────────────────────────
    if _MESSAGING_AVAILABLE:
        _MESSAGING_CFG = _init_messaging(app, {
            "execute_agent":      _execute_agent,
            "register_q":         _register_stream_q,
            "unregister_q":       _unregister_stream_q,
            "get_state":          _get_agent_state,
            "set_state":          _set_agent_state,
            "agent_lock":         _AGENT_LOCK,
            "agent_lock_timeout": _AGENT_LOCK_TIMEOUT,
            "stop_events":        _stop_events,
            "stop_events_lock":   _stop_events_lock,
            "session_lock":       _session_lock,
            "require_auth":       _require_auth,
            "permission_mode":    _resolve_permission_mode,
            "broadcast":          _broadcast,
            "get_messaging_mode": _get_messaging_mode,
            "set_messaging_mode": _set_messaging_mode,
        })
    # ─────────────────────────────────────────────────────────────────────────

    if _SSL_CERT and _SSL_KEY:
        app.run(host=_HOST, port=_PORT, debug=False,
                threaded=True, ssl_context=(_SSL_CERT, _SSL_KEY))
    else:
        app.run(host=_HOST, port=_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
