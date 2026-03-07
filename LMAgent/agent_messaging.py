"""
agent_messaging.py — External messaging integrations for LMAgent
================================================================
Supports: Discord, Telegram, WhatsApp (Green API), Twilio SMS

SETUP
-----
1. Drop this file alongside agent_web.py
2. In agent_web.py add the following (see INTEGRATION GUIDE below)
3. Set env vars for the platforms you want active
4. Install platform deps:
     pip install discord.py python-telegram-bot requests twilio

ENV VARS (only set the ones you want active)
--------------------------------------------
    DISCORD_TOKEN=...
    TELEGRAM_TOKEN=...
    GREEN_API_INSTANCE_ID=...
    GREEN_API_TOKEN=...
    TWILIO_ACCOUNT_SID=...
    TWILIO_AUTH_TOKEN=...
    TWILIO_FROM=+1...           # your Twilio phone number

INTEGRATION GUIDE (changes to agent_web.py)
--------------------------------------------
Near the bottom of agent_web.py, just before main(), add:

    from agent_messaging import init_messaging as _init_messaging
    _MESSAGING_CFG = None   # filled in main()

Inside main(), after _start_web_scheduler(), add:

    global _MESSAGING_CFG
    _MESSAGING_CFG = _init_messaging(app, {
        "execute_agent":       _execute_agent,
        "register_q":          _register_stream_q,
        "unregister_q":        _unregister_stream_q,
        "get_state":           _get_agent_state,
        "set_state":           _set_agent_state,
        "agent_lock":          _AGENT_LOCK,
        "agent_lock_timeout":  _AGENT_LOCK_TIMEOUT,
        "stop_events":         _stop_events,
        "stop_events_lock":    _stop_events_lock,
        "session_lock":        _session_lock,
        "require_auth":        _require_auth,
        "permission_mode":     _resolve_permission_mode,
        "broadcast":           _broadcast,
        "get_messaging_mode":  _get_messaging_mode,
        "set_messaging_mode":  _set_messaging_mode,
    })

PERSISTENT SESSIONS
-------------------
Each user on each platform gets their own persistent session that survives
restarts. Sessions are stored in .lmagent/messaging_sessions.json

Send /new to any bot to wipe your session and start fresh.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, Response, jsonify, request

# ── Load .env file from the same directory as this script ────────────────────
def _load_env_file() -> None:
    _dir = Path(__file__).parent
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
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"[messaging] Warning: could not read env file: {e}")

_load_env_file()
# ─────────────────────────────────────────────────────────────────────────────

# ── Suppress noisy library loggers ───────────────────────────────────────────
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# PERSISTENT SESSION STORE
# =============================================================================
# Maps "platform:sender" → session_id, saved to disk so restarts don't lose
# the conversation. File: <workspace>/.lmagent/messaging_sessions.json

_session_store:       dict           = {}
_session_store_lock:  threading.Lock = threading.Lock()
_session_store_path:  Optional[Path] = None   # set in init_messaging()


def _sessions_path() -> Optional[Path]:
    if _session_store_path:
        return _session_store_path
    ws = os.environ.get("WORKSPACE", "").strip()
    if ws:
        p = Path(ws) / ".lmagent" / "messaging_sessions.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return None


def _load_sessions() -> None:
    p = _sessions_path()
    if not p or not p.exists():
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        with _session_store_lock:
            _session_store.update(data)
        print(f"[messaging] Loaded {len(_session_store)} persistent session(s)")
    except Exception as e:
        print(f"[messaging] Warning: could not load messaging_sessions.json: {e}")


def _save_sessions() -> None:
    p = _sessions_path()
    if not p:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _session_store_lock:
            data = dict(_session_store)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[messaging] Warning: could not save messaging_sessions.json: {e}")


def _get_session(platform: str, sender: str) -> Optional[str]:
    key = f"{platform}:{sender}"
    with _session_store_lock:
        return _session_store.get(key)


def _set_session(platform: str, sender: str, session_id: str) -> None:
    key = f"{platform}:{sender}"
    with _session_store_lock:
        _session_store[key] = session_id
    _save_sessions()


def _clear_session(platform: str, sender: str) -> None:
    key = f"{platform}:{sender}"
    with _session_store_lock:
        _session_store.pop(key, None)
    _save_sessions()


# =============================================================================
# SHARED STATE
# =============================================================================

_active_mode       : str            = "web"
_active_mode_lock  : threading.Lock = threading.Lock()

_platform_status   : dict           = {
    "discord":  "disabled",
    "telegram": "disabled",
    "whatsapp": "disabled",
    "sms":      "disabled",
}
_platform_status_lock: threading.Lock = threading.Lock()

_message_log      : list            = []
_message_log_lock : threading.Lock  = threading.Lock()
_MESSAGE_LOG_MAX  : int             = 50

_cfg: dict = {}

_BUSY_MSG     = "Agent is busy or unavailable right now. Try again in a moment."
_WEB_OWNS_MSG = "Agent is currently in Web mode. Switch to your platform in the LMAgent UI."
_NEW_MSG      = "✓ Starting a fresh session — previous conversation cleared."


# =============================================================================
# HELPERS
# =============================================================================

def _set_platform_status(platform: str, status: str) -> None:
    with _platform_status_lock:
        _platform_status[platform] = status


def _get_mode() -> str:
    with _active_mode_lock:
        return _active_mode


def _set_mode(mode: str) -> None:
    """Set the active messaging mode locally AND notify agent_web so it can
    gate the web SSE stream appropriately."""
    global _active_mode
    with _active_mode_lock:
        _active_mode = mode
    # Propagate to agent_web._messaging_mode so _broadcast() knows whether
    # to suppress content events from the web SSE stream.
    set_web_mode = _cfg.get("set_messaging_mode")
    if set_web_mode:
        set_web_mode(mode)


def _log_message(platform: str, sender: str, text: str,
                 reply: str = "", direction: str = "in") -> None:
    entry = {
        "platform":  platform,
        "sender":    sender,
        "text":      text[:200],
        "reply":     reply[:200] if reply else "",
        "direction": direction,
        "ts":        time.time(),
    }
    with _message_log_lock:
        _message_log.append(entry)
        if len(_message_log) > _MESSAGE_LOG_MAX:
            _message_log.pop(0)


def _can_handle(platform: str) -> tuple[bool, str]:
    mode = _get_mode()
    if mode == "web":
        return False, _WEB_OWNS_MSG
    if mode != platform:
        return False, f"Agent is in {mode.upper()} mode — not accepting {platform} messages."
    state = _cfg.get("get_state", lambda: "idle")()
    if state == "running":
        return False, _BUSY_MSG
    return True, ""


# =============================================================================
# CORE TASK RUNNER
# =============================================================================

def _run_messaging_task(message: str, platform: str,
                        sender: str, reply_fn: Callable[[str], None]) -> None:
    """
    Handles /new to reset session, otherwise resumes the persistent session
    for this sender. Saves the session_id back to disk after each run so
    the next message continues the same conversation — even after a restart.
    """
    # ── /new command — just wipe and confirm, no agent needed ────────────────
    if message.strip().lower() == "/new":
        _clear_session(platform, sender)
        reply_fn(_NEW_MSG)
        return

    execute_agent = _cfg["execute_agent"]
    register_q    = _cfg["register_q"]
    unregister_q  = _cfg["unregister_q"]
    agent_lock    = _cfg["agent_lock"]
    lock_timeout  = _cfg["agent_lock_timeout"]
    stop_events   = _cfg["stop_events"]
    stop_ev_lock  = _cfg["stop_events_lock"]
    perm_mode_fn  = _cfg["permission_mode"]

    _log_message(platform, sender, message, direction="in")

    # Look up this user's persistent session (None = first ever message)
    resume_session = _get_session(platform, sender)

    if not agent_lock.acquire(timeout=lock_timeout):
        reply_fn(_BUSY_MSG)
        return

    request_id = f"msg_{platform}_{secrets.token_hex(6)}"
    stop_ev    = threading.Event()

    with stop_ev_lock:
        stop_events[request_id] = stop_ev

    q           = register_q()
    token_parts = []
    result_ev   = threading.Event()
    final       = [""]
    new_sid     = [resume_session]

    def _on_session(sid: str) -> None:
        """Called directly by _execute_agent with the final session ID."""
        if sid:
            new_sid[0] = sid

    def _listen():
        try:
            while True:
                try:
                    kind, payload = q.get(timeout=300)
                except Exception:
                    final[0] = "Agent timed out."
                    break
                if kind == "token":
                    token_parts.append(payload)

                elif kind == "done":
                    final[0] = "".join(token_parts).strip() or str(payload)
                    break
                elif kind == "error":
                    final[0] = f"Agent error: {payload}"
                    break
        finally:
            unregister_q(q)
            result_ev.set()

    listener = threading.Thread(target=_listen, daemon=True,
                                name=f"msg-listener-{request_id[:8]}")
    listener.start()

    try:
        # PATCH 5: Pass web_origin=False so messaging tasks do not overwrite
        # the browser's _current_session_id or trigger web SSE session events.
        execute_agent(
            message,
            resume_session,
            request_id,
            stop_ev,
            perm_mode_fn(),
            web_origin=False,
            on_session=_on_session,
        )
    except Exception as e:
        final[0] = f"Agent exception: {e}"
        result_ev.set()
    finally:
        agent_lock.release()
        with stop_ev_lock:
            stop_events.pop(request_id, None)

    result_ev.wait(timeout=310)

    # Persist the session_id so the next message resumes it
    if new_sid[0]:
        _set_session(platform, sender, new_sid[0])

    answer = final[0] or "No response generated."
    _log_message(platform, sender, message, reply=answer, direction="out")
    reply_fn(answer)


# =============================================================================
# DISCORD INTEGRATION
# =============================================================================

def _start_discord(token: str) -> None:
    try:
        import discord
    except ImportError:
        print("[messaging] discord.py not installed — run: pip install discord.py")
        _set_platform_status("discord", "error")
        return

    intents                 = discord.Intents.default()
    intents.message_content = True
    client                  = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[messaging/discord] Connected as {client.user}")
        _set_platform_status("discord", "connected")

    @client.event
    async def on_message(msg):
        if msg.author == client.user:
            return
        is_dm      = isinstance(msg.channel, discord.DMChannel)
        is_mention = client.user in msg.mentions
        if not is_dm and not is_mention:
            return

        text = msg.content.replace(f"<@{client.user.id}>", "").strip()
        if not text:
            return

        # /new bypasses the mode check — always allowed
        if text.strip().lower() != "/new":
            ok, reason = _can_handle("discord")
            if not ok:
                await msg.reply(reason)
                return

        async with msg.channel.typing():
            result_holder = [None]
            done_ev       = threading.Event()

            def _reply(answer: str):
                result_holder[0] = answer
                done_ev.set()

            t = threading.Thread(
                target=_run_messaging_task,
                args=(text, "discord", str(msg.author), _reply),
                daemon=True,
            )
            t.start()

            import asyncio
            while not done_ev.wait(timeout=0.5):
                await asyncio.sleep(0.5)

        answer = result_holder[0] or _BUSY_MSG
        for chunk in _chunk(answer, 1900):
            await msg.reply(chunk)

    @client.event
    async def on_disconnect():
        _set_platform_status("discord", "disconnected")

    def _run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(client.start(token))
        except Exception as e:
            print(f"[messaging/discord] Error: {e}")
            _set_platform_status("discord", "error")

    _set_platform_status("discord", "disconnected")
    t = threading.Thread(target=_run, daemon=True, name="discord-bot")
    t.start()
    print("[messaging] Discord bot thread started")


# =============================================================================
# TELEGRAM INTEGRATION
# =============================================================================

def _start_telegram(token: str) -> None:
    try:
        from telegram import Update
        from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
    except ImportError:
        print("[messaging] python-telegram-bot not installed — run: pip install python-telegram-bot")
        _set_platform_status("telegram", "error")
        return

    async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        text   = update.message.text.strip()
        sender = str(update.effective_user.id) if update.effective_user else "unknown"

        if text.strip().lower() != "/new":
            ok, reason = _can_handle("telegram")
            if not ok:
                await update.message.reply_text(reason)
                return

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )

        result_holder = [None]
        done_ev       = threading.Event()

        def _reply(answer: str):
            result_holder[0] = answer
            done_ev.set()

        t = threading.Thread(
            target=_run_messaging_task,
            args=(text, "telegram", sender, _reply),
            daemon=True,
        )
        t.start()
        done_ev.wait(timeout=300)

        answer = result_holder[0] or _BUSY_MSG
        for chunk in _chunk(answer, 4000):
            await update.message.reply_text(chunk)

    async def _new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sender = str(update.effective_user.id) if update.effective_user else "unknown"
        _clear_session("telegram", sender)
        await update.message.reply_text(_NEW_MSG)

    def _run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            app = Application.builder().token(token).build()
            app.add_handler(CommandHandler("new", _new_cmd))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle))
            _set_platform_status("telegram", "connected")
            print("[messaging/telegram] Bot polling started")
            loop.run_until_complete(app.run_polling(drop_pending_updates=True))
        except Exception as e:
            print(f"[messaging/telegram] Error: {e}")
            _set_platform_status("telegram", "error")

    _set_platform_status("telegram", "disconnected")
    t = threading.Thread(target=_run, daemon=True, name="telegram-bot")
    t.start()
    print("[messaging] Telegram bot thread started")


# =============================================================================
# WHATSAPP — GREEN API
# =============================================================================

def _start_whatsapp(instance_id: str, api_token: str) -> None:
    BASE = f"https://api.green-api.com/waInstance{instance_id}"
    RECV = f"{BASE}/receiveNotification/{api_token}"
    DEL  = f"{BASE}/deleteNotification/{api_token}"

    def _send_reply(chat_id: str, text: str) -> None:
        import requests as _req
        url = f"{BASE}/sendMessage/{api_token}"
        for chunk in _chunk(text, 4096):
            try:
                _req.post(url, json={"chatId": chat_id, "message": chunk}, timeout=15)
            except Exception as e:
                print(f"[messaging/whatsapp] send error: {e}")

    def _poll_loop():
        import requests as _req
        _set_platform_status("whatsapp", "connected")
        print("[messaging/whatsapp] Polling Green API…")
        while True:
            try:
                resp = _req.get(RECV, timeout=35)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        receipt_id = data.get("receiptId")
                        body       = data.get("body", {})
                        msg_type   = body.get("typeWebhook", "")

                        if msg_type == "incomingMessageReceived":
                            msg_data = body.get("messageData", {})
                            text_msg = msg_data.get("textMessageData", {})
                            text     = text_msg.get("textMessage", "").strip()
                            sender   = body.get("senderData", {}).get("sender", "unknown")
                            chat_id  = body.get("senderData", {}).get("chatId", sender)

                            if text:
                                if text.strip().lower() != "/new":
                                    ok, reason = _can_handle("whatsapp")
                                    if not ok:
                                        _send_reply(chat_id, reason)
                                        if receipt_id is not None:
                                            try:
                                                _req.delete(f"{DEL}/{receipt_id}", timeout=10)
                                            except Exception:
                                                pass
                                        continue

                                result_holder = [None]
                                done_ev       = threading.Event()

                                def _reply(answer: str, cid=chat_id):
                                    result_holder[0] = answer
                                    done_ev.set()

                                t = threading.Thread(
                                    target=_run_messaging_task,
                                    args=(text, "whatsapp", sender, _reply),
                                    daemon=True,
                                )
                                t.start()
                                done_ev.wait(timeout=300)
                                _send_reply(chat_id, result_holder[0] or _BUSY_MSG)

                        if receipt_id is not None:
                            try:
                                _req.delete(f"{DEL}/{receipt_id}", timeout=10)
                            except Exception:
                                pass

                elif resp.status_code == 401:
                    print("[messaging/whatsapp] Auth error — check instance ID / token")
                    _set_platform_status("whatsapp", "error")
                    break
            except Exception as e:
                print(f"[messaging/whatsapp] poll error: {e}")
                _set_platform_status("whatsapp", "disconnected")
                time.sleep(5)
                _set_platform_status("whatsapp", "connected")

    _set_platform_status("whatsapp", "disconnected")
    t = threading.Thread(target=_poll_loop, daemon=True, name="whatsapp-poll")
    t.start()
    print("[messaging] WhatsApp (Green API) poll thread started")


# =============================================================================
# TWILIO SMS INTEGRATION
# =============================================================================

def _register_twilio_webhook(app: Flask, account_sid: str,
                              auth_token: str, from_number: str) -> None:
    try:
        from twilio.request_validator import RequestValidator
        _validator      = RequestValidator(auth_token)
        _has_twilio_lib = True
    except ImportError:
        print("[messaging] twilio library not installed — run: pip install twilio")
        print("[messaging/sms] WARNING: requests will NOT be signature-validated")
        _has_twilio_lib = False
        _validator      = None

    _set_platform_status("sms", "connected")
    print(f"[messaging/sms] Webhook registered at /sms  (from {from_number})")

    @app.route("/sms", methods=["POST"])
    def sms_webhook():
        if _has_twilio_lib and _validator:
            sig  = request.headers.get("X-Twilio-Signature", "")
            url  = request.url
            post = request.form.to_dict()
            if not _validator.validate(url, post, sig):
                return Response("Forbidden", status=403)

        text   = request.form.get("Body", "").strip()
        sender = request.form.get("From", "unknown")

        if not text:
            return _twiml("")

        if text.strip().lower() != "/new":
            ok, reason = _can_handle("sms")
            if not ok:
                return _twiml(reason)

        result_holder = [None]
        done_ev       = threading.Event()

        def _reply(answer: str):
            result_holder[0] = answer
            done_ev.set()

        t = threading.Thread(
            target=_run_messaging_task,
            args=(text, "sms", sender, _reply),
            daemon=True,
        )
        t.start()
        done_ev.wait(timeout=295)

        answer = result_holder[0] or _BUSY_MSG
        return _twiml(answer[:1590])


def _twiml(text: str) -> Response:
    body = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml  = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{body}</Message></Response>'
    return Response(xml, mimetype="text/xml")


# =============================================================================
# UTILITY
# =============================================================================

def _chunk(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    parts, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > size:
            if buf:
                parts.append(buf)
            buf = ""
        buf += line
    if buf:
        parts.append(buf)
    return parts or [text[:size]]


# =============================================================================
# FLASK ROUTES
# =============================================================================

def _register_routes(app: Flask, require_auth) -> None:

    @app.route("/messaging/status")
    @require_auth
    def messaging_status():
        with _platform_status_lock:
            statuses = dict(_platform_status)
        with _message_log_lock:
            recent = list(_message_log[-20:])
        return jsonify({
            "mode":      _get_mode(),
            "platforms": statuses,
            "recent":    recent,
        })

    @app.route("/messaging/mode", methods=["POST"])
    @require_auth
    def messaging_set_mode():
        data = request.get_json(force=True, silent=True) or {}
        mode = (data.get("mode") or "web").lower()
        valid = {"web", "discord", "telegram", "whatsapp", "sms"}
        if mode not in valid:
            return jsonify({"error": f"invalid mode: {mode}"}), 400
        # _set_mode also calls cfg["set_messaging_mode"] to update the gate
        # inside agent_web._broadcast, so content events are suppressed from
        # the web SSE stream whenever a non-web platform is active.
        _set_mode(mode)
        return jsonify({"ok": True, "mode": mode})

    @app.route("/messaging/feed")
    @require_auth
    def messaging_feed():
        n = min(int(request.args.get("n", 20)), 50)
        with _message_log_lock:
            recent = list(_message_log[-n:])
        return jsonify({"entries": recent})

    @app.route("/messaging.js")
    def messaging_js():
        token = (request.headers.get("X-Token", "")
                 or request.args.get("token", ""))
        try:
            with open("agent_messaging_ui.js", "r", encoding="utf-8") as f:
                src = f.read()
        except FileNotFoundError:
            return Response("// agent_messaging_ui.js not found", mimetype="application/javascript")
        src = src.replace("__AGENT_TOKEN__", token)
        return Response(src, mimetype="application/javascript")


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def init_messaging(app: Flask, cfg: dict) -> dict:
    global _cfg, _session_store_path

    _cfg = cfg

    # Resolve workspace path for session file
    try:
        from agent_core import Config
        ws = Path(Config.WORKSPACE)
    except Exception:
        ws = Path(os.environ.get("WORKSPACE", "."))

    _session_store_path = ws / ".lmagent" / "messaging_sessions.json"
    _session_store_path.parent.mkdir(parents=True, exist_ok=True)
    _load_sessions()

    require_auth = cfg["require_auth"]
    _register_routes(app, require_auth)

    # ── Discord ───────────────────────────────────────────────────────────────
    discord_token = os.environ.get("DISCORD_TOKEN", "").strip()
    if discord_token:
        _start_discord(discord_token)
    else:
        print("[messaging] DISCORD_TOKEN not set — Discord disabled")

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    if telegram_token:
        _start_telegram(telegram_token)
    else:
        print("[messaging] TELEGRAM_TOKEN not set — Telegram disabled")

    # ── WhatsApp (Green API) ──────────────────────────────────────────────────
    green_instance = os.environ.get("GREEN_API_INSTANCE_ID", "").strip()
    green_token    = os.environ.get("GREEN_API_TOKEN", "").strip()
    if green_instance and green_token:
        _start_whatsapp(green_instance, green_token)
    else:
        print("[messaging] GREEN_API_INSTANCE_ID / GREEN_API_TOKEN not set — WhatsApp disabled")

    # ── Twilio SMS ────────────────────────────────────────────────────────────
    twilio_sid  = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    twilio_from = os.environ.get("TWILIO_FROM", "").strip()
    if twilio_sid and twilio_auth and twilio_from:
        _register_twilio_webhook(app, twilio_sid, twilio_auth, twilio_from)
    else:
        print("[messaging] TWILIO_* vars not set — SMS disabled")

    print("[messaging] init_messaging() complete")
    return _cfg
