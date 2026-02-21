#!/usr/bin/env python3
"""
agent_core.py — Foundation layer for LM Agent.

Changes:
  - PowerShell sessions are now per-thread (thread-local) instead of a global
    singleton — concurrent scheduler sessions no longer share one PS process.
  - Log._silent is now thread-local — output-mode sessions no longer silently
    suppress logging in concurrent interactive sessions.
  - SessionManager.create() appends a uuid4 suffix so sub-agents spawned in
    the same second no longer silently overwrite each other's session files.
  - ProgressTracker (dead code — format it expected was never emitted) removed;
    _build_progress_summary() now reads from actual TaskState messages and real
    tool results instead of always returning "No measurable progress".
  - compact_messages() threshold now reads Config.SUMMARIZATION_THRESHOLD
    instead of the hardcoded 80 000 that made the env-var do nothing.
  - compact_messages() keeps only the LATEST plan/todo injection message
    instead of re-appending every historical one (which caused duplicates that
    compounded with the ones _inject_context() re-inserts every iteration).
  - close_shell_session() exported so callers (agent_main finally block)
    can clean up the current thread's shell process explicitly.

Cross-platform changes:
  - .env file loader added at module load time — users can set config in a
    plain text file instead of system environment variables.
  - PersistentPowerShellSession replaced with cross-platform ShellSession:
    uses powershell.exe on Windows, /bin/bash on Linux/macOS.
    Command wrapping, exit-code capture, and init commands adapt per-platform.
  - InstanceLock no longer uses ctypes.windll — uses os.kill(pid, 0) on
    non-Windows systems to check whether a PID is alive.
  - subprocess.CREATE_NO_WINDOW flag is only applied on Windows.

"""
import atexit
import difflib
import json
import math
import hashlib
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' required.  pip install requests")

try:
    import colorama
    colorama.init()
except ImportError:
    pass

VERSION = "9.3.1-cross-platform"

_IS_WINDOWS = platform.system() == "Windows"


# =============================================================================
# .ENV FILE LOADER
# Loaded before Config so env-var defaults pick up the values.
# Searches: <script dir>/.env, then cwd/.env. Does NOT override existing vars.
# =============================================================================

def _load_dotenv():
    """Load key=value pairs from a .env file into os.environ.

    Checks (in order):
      1. Directory containing this script
      2. Current working directory
    Existing environment variables are never overridden.
    Lines starting with # and blank lines are ignored.
    Values may be optionally quoted with single or double quotes.
    """
    candidates = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_file in candidates:
        if not env_file.exists():
            continue
        try:
            loaded = 0
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                # Strip surrounding quotes (single or double)
                if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded += 1
            if loaded:
                # Use print() here — Log class not yet defined
                print(f"[INFO] Loaded {loaded} variable(s) from {env_file}")
        except Exception as e:
            print(f"[!] Could not read {env_file}: {e}")
        break  # stop after first found


_load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

class PermissionMode(Enum):
    MANUAL = "manual"
    NORMAL = "normal"
    AUTO   = "auto"


class Config:
    """Central config — every value overridable via environment variable."""

    # LLM
    LLM_URL         = os.getenv("LLM_URL",         "http://localhost:1234/v1/chat/completions")
    LLM_API_KEY     = os.getenv("LLM_API_KEY",     "lm-studio")
    LLM_MODEL       = os.getenv("LLM_MODEL",       "")
    MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS",    "-1"))
    TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.9"))

    # Context management
    SUMMARIZATION_THRESHOLD = int(os.getenv("SUMMARIZATION_THRESHOLD", "80000"))
    KEEP_RECENT_MESSAGES    = int(os.getenv("KEEP_RECENT_MESSAGES",    "30"))

    # Retry / timeouts
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "3.0"))
    LLM_TIMEOUT     = int(os.getenv("LLM_TIMEOUT",     "560"))

    # Workspace
    WORKSPACE = os.getenv("WORKSPACE", str(Path.home() / "lm_workspace"))

    # Execution limits
    MAX_ITERATIONS           = int(os.getenv("MAX_ITERATIONS",           "150"))
    MAX_SUB_AGENT_ITERATIONS = int(os.getenv("MAX_SUB_AGENT_ITERATIONS", "25"))
    MAX_SAME_TOOL_STREAK     = int(os.getenv("MAX_SAME_TOOL_STREAK",     "8"))
    MAX_NO_PROGRESS_ITERS    = int(os.getenv("MAX_NO_PROGRESS_ITERS",    "15"))
    MAX_ERRORS               = int(os.getenv("MAX_ERRORS",               "5"))
    MAX_EMPTY_ITERATIONS     = int(os.getenv("MAX_EMPTY_ITERATIONS",     "5"))

    # Tool output limits
    MAX_TOOL_OUTPUT  = int(os.getenv("MAX_TOOL_OUTPUT",  "500000"))
    MAX_FILE_READ    = int(os.getenv("MAX_FILE_READ",    "1000000"))
    MAX_GREP_RESULTS = int(os.getenv("MAX_GREP_RESULTS", "50"))
    MAX_LS_ENTRIES   = int(os.getenv("MAX_LS_ENTRIES",   "100"))

    # Safety
    REQUIRE_WORKSPACE = os.getenv("REQUIRE_WORKSPACE", "true").lower() == "true"
    BLOCKED_COMMANDS  = os.getenv(
        "BLOCKED_COMMANDS",
        r"rmdir /s /q C:\,format,del /f /s /q C:\,rm -rf /,mkfs" if _IS_WINDOWS
        else "rm -rf /,mkfs,dd if=/dev/zero of=/dev/sd"
    ).split(",")

    # Feature flags
    PERMISSION_MODE         = os.getenv("PERMISSION_MODE",         "normal")
    ENABLE_MCP              = os.getenv("ENABLE_MCP",              "true").lower() == "true"
    AUTO_SAVE_SESSION       = os.getenv("AUTO_SAVE_SESSION",       "true").lower() == "true"
    ENABLE_SUMMARIZATION    = os.getenv("ENABLE_SUMMARIZATION",    "true").lower() == "true"
    ENABLE_SUB_AGENTS       = os.getenv("ENABLE_SUB_AGENTS",       "true").lower() == "true"
    ENABLE_TODO_TRACKING    = os.getenv("ENABLE_TODO_TRACKING",    "true").lower() == "true"
    ENABLE_PLAN_ENFORCEMENT = os.getenv("ENABLE_PLAN_ENFORCEMENT", "true").lower() == "true"

    # Scheduler
    SCHEDULER_POLL_INTERVAL = int(os.getenv("SCHEDULER_POLL_INTERVAL", "60"))

    # Thinking-model support (QwQ, DeepSeek-R1, etc.)
    THINKING_MODEL      = os.getenv("THINKING_MODEL", "true").lower() == "true"
    THINKING_MAX_TOKENS = int(os.getenv("THINKING_MAX_TOKENS", "16000"))

    # Binary extensions — never read/write as text
    BINARY_EXTS = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip",
        ".gz", ".tar", ".exe", ".dll", ".so", ".pyc", ".bin", ".dat",
        ".mp4", ".mp3", ".avi", ".mov", ".iso", ".dmg",
    })

    READ_ONLY_TOOLS   = frozenset({"read", "ls", "glob", "grep",
                                    "git_status", "git_diff", "todo_list"})
    DESTRUCTIVE_TOOLS = frozenset({"write", "edit", "shell", "git_add",
                                    "git_commit", "git_branch",
                                    "todo_add", "todo_update", "todo_complete"})

    @classmethod
    def init(cls):
        workspace = Path(cls.WORKSPACE)
        for sub in ("sessions", "locks", "todos", "plans", "state", "inbox"):
            (workspace / ".lmagent" / sub).mkdir(parents=True, exist_ok=True)
        cls._validate()
        if cls.THINKING_MODEL:
            Log.info("Thinking model mode: ON (reasoning tokens stripped from context)")

    @classmethod
    def _validate(cls):
        if cls.MAX_ITERATIONS < 1:
            raise ValueError("MAX_ITERATIONS must be >= 1")
        try:
            PermissionMode(cls.PERMISSION_MODE)
        except ValueError:
            raise ValueError(f"Invalid PERMISSION_MODE: {cls.PERMISSION_MODE}")


# =============================================================================
# COLORS & LOGGING
# =============================================================================

class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[38;5;196m"
    GREEN   = "\033[38;5;202m"
    YELLOW  = "\033[38;5;214m"
    BLUE    = "\033[38;5;223m"
    MAGENTA = "\033[38;5;208m"
    CYAN    = "\033[38;5;215m"
    GRAY    = "\033[38;5;250m"


def _supports_truecolor() -> bool:
    forced = os.environ.get("FORCE_TRUECOLOR")
    if forced == "1": return True
    if forced == "0": return False
    try:
        if not sys.stdout.isatty(): return False
    except Exception:
        return False
    colorterm    = os.environ.get("COLORTERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    return ("truecolor" in colorterm or "24bit" in colorterm or
            any(x in term_program for x in
                ("wezterm", "alacritty", "iterm", "vscode", "windows_terminal")))


def _rgb_escape(r: int, g: int, b: int, bold: bool = False) -> str:
    r, g, b = (max(0, min(255, int(x))) for x in (r, g, b))
    return f"{Colors.BOLD if bold else ''}\033[38;2;{r};{g};{b}m"


def _rainbow_rgb(i: int, n: int, phase: float = 0.0) -> Tuple[int, int, int]:
    t = 0.0 if n <= 1 else (i / (n - 1)) * math.pi
    return (
        max(0, min(255, int(140 + 100 * math.sin(t + phase)))),
        max(0, min(255, int(30  + 40  * math.sin(t + phase)))),
        max(0, min(255, int(180 + 75  * math.sin(t + phase)))),
    )


def colored(text: str, color: str, bold: bool = False) -> str:
    return f"{Colors.BOLD if bold else ''}{color}{text}{Colors.RESET}"


def rainbow_text(text: str, bold: bool = False, phase: float = 0.0,
                 fallback_color: str = None) -> str:
    fallback_color = fallback_color or Colors.GREEN
    try:
        if not _supports_truecolor():
            return colored(text, fallback_color, bold=bold)
        paintable = [c for c in text if c.strip()]
        n = len(paintable)
        out, idx = [], 0
        for c in text:
            if c.strip():
                r, g, b = _rainbow_rgb(idx, n, phase=phase)
                out.append(f"{_rgb_escape(r, g, b, bold=bold)}{c}{Colors.RESET}")
                idx += 1
            else:
                out.append(c)
        return "".join(out)
    except Exception:
        return colored(text, fallback_color, bold=bold)


# Log._silent is thread-local — concurrent sessions don't interfere.
_log_local = threading.local()


class Log:
    """Coloured logger. Call Log.set_silent(True) in output/service mode.
    Silent state is per-thread — concurrent sessions don't interfere."""

    @classmethod
    def set_silent(cls, silent: bool):
        _log_local.silent = silent

    @staticmethod
    def _is_silent() -> bool:
        return getattr(_log_local, "silent", False)

    @staticmethod
    def _print(prefix: str, msg: str, color: str):
        if not Log._is_silent():
            print(colored(f"{prefix} {msg}", color))

    @staticmethod
    def info(msg: str):    Log._print("[INFO]", msg, Colors.CYAN)
    @staticmethod
    def success(msg: str): Log._print("[✓]",    msg, Colors.GREEN)
    @staticmethod
    def warning(msg: str): Log._print("[!]",    msg, Colors.YELLOW)
    @staticmethod
    def error(msg: str):   Log._print("[✗]",    msg, Colors.RED)
    @staticmethod
    def tool(name: str, args: str):
        if not Log._is_silent():
            print(colored(f"[→] {name}({args})", Colors.MAGENTA))
    @staticmethod
    def plan(msg: str): Log._print("[📋]", msg, Colors.BLUE)
    @staticmethod
    def task(msg: str): Log._print("[🎯]", msg, Colors.YELLOW)
    @staticmethod
    def wait(msg: str): Log._print("[⏸️]",  msg, Colors.MAGENTA)


# =============================================================================
# UTILITIES
# =============================================================================

def truncate_output(text: str, max_length: int, label: str = "output") -> str:
    if len(text) <= max_length:
        return text
    half        = max_length // 2
    total_lines = text.count("\n") + 1
    return (text[:half]
            + f"\n\n... [TRUNCATED {len(text) - max_length} chars,"
              f" {total_lines} total lines] ...\n\n"
            + text[-(max_length - half - 100):])


_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)


def strip_thinking(content: str) -> Tuple[str, str]:
    """Remove <think>…</think> blocks. Returns (clean_content, thinking_text)."""
    if '<think>' not in content.lower():
        return content, ''
    parts = _THINK_RE.findall(content)
    return _THINK_RE.sub('', content).strip(), '\n'.join(parts)


def _atomic_write(path: Path, data: Dict[str, Any]):
    """Write JSON atomically via a .tmp sibling."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    path.unlink(missing_ok=True)
    tmp.rename(path)


# =============================================================================
# THREAD-LOCAL CONTEXT
# =============================================================================

_ctx_local = threading.local()


def get_current_context() -> Dict[str, Any]:
    return getattr(_ctx_local, "context", {})


def set_current_context(ctx: Dict[str, Any]):
    _ctx_local.context = ctx


def _get_ctx(key: str, label: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Return (manager, None) on success, or (None, error_dict) on failure."""
    mgr = get_current_context().get(key)
    if mgr is None:
        return None, {"success": False, "error": f"{label} unavailable"}
    return mgr, None


# =============================================================================
# INSTANCE LOCK  (cross-platform)
# =============================================================================

class InstanceLock:
    def __init__(self, workspace: Path):
        self.lockfile = workspace / ".lmagent" / "locks" / "agent.lock"
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        self.locked = False

    def _pid_alive(self, pid: int) -> bool:
        """Return True if *pid* is a running process on this machine."""
        if _IS_WINDOWS:
            # Windows: open a handle with SYNCHRONIZE access (0x100000);
            # a zero handle means the PID does not exist.
            try:
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, pid)
                if handle == 0:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            except Exception:
                # If ctypes fails for any reason, fall through to the
                # mtime-based heuristic below.
                return False
        else:
            # POSIX: sending signal 0 checks existence without actually
            # delivering a signal.
            try:
                os.kill(pid, 0)
                return True          # process exists (and we can signal it)
            except ProcessLookupError:
                return False         # no such PID
            except PermissionError:
                return True          # exists but owned by another user

    def acquire(self) -> bool:
        if self.lockfile.exists():
            stale = False
            try:
                pid   = int(self.lockfile.read_text().strip())
                stale = not self._pid_alive(pid)
                if not stale:
                    # Extra safety: treat the lock as stale if the file is
                    # more than 5 minutes old regardless of PID status.
                    if time.time() - self.lockfile.stat().st_mtime > 300:
                        stale = True
            except (ValueError, OSError):
                stale = True

            if stale:
                Log.warning("Removing stale lock")
                self.lockfile.unlink(missing_ok=True)
            else:
                return False

        try:
            self.lockfile.write_text(str(os.getpid()))
            self.locked = True
            return True
        except Exception as e:
            Log.error(f"Failed to acquire lock: {e}")
            return False

    def release(self):
        if self.locked:
            self.lockfile.unlink(missing_ok=True)
            self.locked = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Another agent instance is already running")
        return self

    def __exit__(self, *_):
        self.release()


# =============================================================================
# CLEANUP
# =============================================================================

_instance_lock: Optional[InstanceLock] = None


def cleanup_resources():
    global _instance_lock
    close_shell_session()   # closes THIS thread's shell session
    if _instance_lock:
        try:
            _instance_lock.release()
        except Exception:
            pass
        _instance_lock = None


atexit.register(cleanup_resources)


# =============================================================================
# TOKEN COUNTING
# =============================================================================

class TokenCounter:
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return int(len(text) / 3.5)

    @staticmethod
    def count_message_tokens(msg: Dict[str, Any]) -> int:
        total  = TokenCounter.estimate_tokens(str(msg.get("content") or ""))
        total += sum(TokenCounter.estimate_tokens(json.dumps(tc))
                     for tc in (msg.get("tool_calls") or []))
        return total + 10

    @staticmethod
    def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
        return sum(TokenCounter.count_message_tokens(m) for m in messages)


# =============================================================================
# CROSS-PLATFORM SHELL SESSION
# =============================================================================
#
# On Windows  → powershell.exe  (identical behaviour to the old class)
# On Linux/macOS → /bin/bash
#
# Both variants:
#   • run a long-lived subprocess with stdin/stdout pipes
#   • wrap each command so the exit code is captured and echoed as a
#     recognisable sentinel line  "__DONE_N__:<exit_code>"
#   • use a background reader thread + Queue to avoid blocking
#
# Command-wrapping differs per platform:
#
#   PowerShell:
#       try {
#           <command>
#           $exitCode = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 0 }
#       } catch {
#           Write-Host $_.Exception.Message
#           $exitCode = 1
#       }
#       Write-Host "__DONE_N__:$exitCode"
#
#   Bash:
#       (
#           <command>
#       )
#       _ec=$?
#       echo "__DONE_N__:$_ec"
#
# =============================================================================

class ShellSession:
    """Long-lived shell process with a persistent stdout-reader thread.

    Automatically selects the right shell for the current platform:
      • Windows  → powershell.exe
      • Linux / macOS → /bin/bash
    """

    def __init__(self, workspace: Path):
        self.workspace     = workspace
        self.process: Optional[subprocess.Popen] = None
        self.command_count = 0
        self._out_queue: queue.Queue = queue.Queue()
        self._reader:    Optional[threading.Thread] = None
        self._start_session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _popen_kwargs(workspace: Path) -> Dict[str, Any]:
        """Return platform-appropriate Popen keyword arguments."""
        kwargs: Dict[str, Any] = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workspace),
            text=True,
            bufsize=1,
        )
        if _IS_WINDOWS:
            # CREATE_NO_WINDOW suppresses the console window on Windows.
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kwargs

    @staticmethod
    def _shell_argv() -> List[str]:
        if _IS_WINDOWS:
            return ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", "-"]
        return ["/bin/bash"]

    @staticmethod
    def _init_commands(workspace: Path) -> List[str]:
        """Commands sent once at session start to configure the shell."""
        if _IS_WINDOWS:
            return [
                f"Set-Location '{workspace}'",
                "$ErrorActionPreference = 'Continue'",
            ]
        # bash: cd to workspace; disable history to avoid .bash_history noise
        return [
            f"cd '{workspace}'",
            "export HISTFILE=/dev/null",
        ]

    def _wrap_command(self, command: str, marker: str) -> str:
        """Wrap *command* so the shell echoes the marker and exit code when done."""
        if _IS_WINDOWS:
            return (
                f"try {{\n    {command}\n"
                f"    $exitCode = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} else {{ 0 }}\n"
                f"}} catch {{\n    Write-Host $_.Exception.Message\n    $exitCode = 1\n}}\n"
                f'Write-Host "{marker}:$exitCode"\n'
            )
        # Bash: run in a subshell so the outer script keeps running even if
        # the inner command calls exit.
        return (
            f"(\n{command}\n)\n"
            f"_ec=$?\n"
            f'echo "{marker}:$_ec"\n'
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _start_session(self):
        try:
            self.process = subprocess.Popen(
                self._shell_argv(),
                **self._popen_kwargs(self.workspace),
            )
            self._out_queue = queue.Queue()
            self._reader    = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

            for cmd in self._init_commands(self.workspace):
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            time.sleep(0.5)
        except Exception as e:
            Log.error(f"Failed to start shell session: {e}")
            self.process = None

    def _read_loop(self):
        try:
            for line in self.process.stdout:
                self._out_queue.put(line.rstrip("\n\r"))
        except Exception:
            pass
        finally:
            self._out_queue.put(None)  # sentinel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, command: str, timeout: int = 60) -> Tuple[str, int]:
        if not self.process or self.process.poll() is not None:
            Log.warning("Shell session died, restarting…")
            self._start_session()
            if not self.process:
                return "ERROR: Could not start shell session", 1

        marker  = f"__DONE_{self.command_count}__"
        self.command_count += 1
        wrapped = self._wrap_command(command, marker)

        try:
            self.process.stdin.write(wrapped + "\n")
            self.process.stdin.flush()
        except Exception as e:
            return f"ERROR: {e}", 1

        output_lines: List[str] = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return f"ERROR: Timeout after {timeout}s", 124
            try:
                line = self._out_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                return "ERROR: Shell session terminated", 1
            if line.startswith(marker + ":"):
                try:
                    return_code = int(line.split(":", 1)[1].strip())
                except ValueError:
                    return_code = 1
                return "\n".join(output_lines), return_code
            output_lines.append(line)

    def close(self):
        if self.process:
            try:
                if _IS_WINDOWS:
                    self.process.stdin.write("exit\n")
                else:
                    self.process.stdin.write("exit 0\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None


# ---------------------------------------------------------------------------
# Backward-compat alias — agent_tools.py and any external caller can still
# reference PersistentPowerShellSession; it now resolves to ShellSession.
# ---------------------------------------------------------------------------
PersistentPowerShellSession = ShellSession


# Thread-local storage — each agent thread gets its own shell process.
_shell_local = threading.local()


def get_shell_session(workspace: Path) -> ShellSession:
    """Return this thread's shell session, creating one if needed.
    Each agent thread gets an isolated process — no cross-session interference."""
    session: Optional[ShellSession] = getattr(_shell_local, "session", None)
    if session is None:
        session = ShellSession(workspace)
        _shell_local.session = session
    return session


def close_shell_session():
    """Close and discard this thread's shell session.
    Call this in the finally block of run_agent() for clean resource release.
    The main thread's session is also closed by the atexit handler."""
    session: Optional[ShellSession] = getattr(_shell_local, "session", None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
        _shell_local.session = None


# Backward-compat aliases used in agent_tools.py
def get_powershell_session(workspace: Path) -> ShellSession:
    return get_shell_session(workspace)


def close_powershell_session():
    close_shell_session()


# =============================================================================
# FILE EDITOR
# =============================================================================

class FileEditor:
    @staticmethod
    def _normalize(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.split("\n"))

    @staticmethod
    def _fuzzy_search(content: str, search: str,
                      threshold: float = 0.75) -> Optional[Tuple[int, int, float]]:
        content_lines = content.split("\n")
        search_lines  = search.split("\n")
        best_ratio    = 0.0
        best: Optional[Tuple[int, int, float]] = None
        for i in range(len(content_lines) - len(search_lines) + 1):
            block = content_lines[i: i + len(search_lines)]
            ratio = difflib.SequenceMatcher(None, search_lines, block).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                start = sum(len(l) + 1 for l in content_lines[:i])
                end   = start + sum(len(l) + 1 for l in block)
                best  = (start, end - 1, ratio)
        return best

    @staticmethod
    def search_replace(content: str, search: str,
                       replace: str) -> Tuple[bool, str, str]:
        # 1. Exact
        count = content.count(search)
        if count == 1:
            return True, content.replace(search, replace), "Exact match"
        if count > 1:
            return False, content, f"Search appears {count} times (must be unique)"

        # 2. Whitespace-normalised
        norm_content = FileEditor._normalize(content)
        norm_search  = FileEditor._normalize(search)
        if norm_search in norm_content:
            idx            = norm_content.index(norm_search)
            lines_before   = norm_content[:idx].count("\n")
            content_lines  = content.split("\n")
            search_n_lines = len(search.split("\n"))
            for i in range(max(0, lines_before - 2),
                           min(len(content_lines), lines_before + 3)):
                block = "\n".join(content_lines[i: i + search_n_lines])
                if FileEditor._normalize(block) == norm_search:
                    new = "\n".join(
                        content_lines[:i]
                        + replace.split("\n")
                        + content_lines[i + search_n_lines:]
                    )
                    return True, new, "Whitespace-insensitive match"

        # 3. Fuzzy
        match = FileEditor._fuzzy_search(content, search)
        if match:
            start, end, ratio = match
            return (True,
                    content[:start] + replace + content[end + 1:],
                    f"Fuzzy match ({ratio * 100:.0f}%)")

        preview = "\n".join(content.split("\n")[:5])[:200]
        return False, content, f"Not found. File preview:\n{preview}…"


# =============================================================================
# SAFETY
# =============================================================================

class Safety:
    _SENSITIVE_WINDOWS = (
        "\\Windows\\System32", "C:\\Windows", "\\Program Files",
        "\\Users\\All Users", "\\ProgramData",
        "id_rsa", "id_ed25519", ".pem", ".key",
    )
    _SENSITIVE_POSIX = (
        "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        "/root/", "/boot/", "/sys/",
        "id_rsa", "id_ed25519", ".pem", ".key",
    )
    _SENSITIVE = _SENSITIVE_WINDOWS if _IS_WINDOWS else _SENSITIVE_POSIX

    @staticmethod
    def validate_path(workspace: Path, path: str,
                      must_exist: bool = False) -> Tuple[bool, str, Path]:
        try:
            p        = Path(path)
            resolved = ((workspace / path) if not p.is_absolute()
                        else p.expanduser()).resolve()
            if must_exist and not resolved.exists():
                return False, f"Path does not exist: {path}", resolved
            if Config.REQUIRE_WORKSPACE:
                try:
                    resolved.relative_to(workspace.resolve())
                except ValueError:
                    return False, f"Path outside workspace: {path}", resolved
            for s in Safety._SENSITIVE:
                if s.lower() in str(resolved).lower():
                    return False, f"Access denied: {path}", resolved
            return True, "", resolved
        except Exception as e:
            return False, f"Invalid path: {e}", Path(path)

    @staticmethod
    def validate_command(cmd: str) -> Tuple[bool, str]:
        cmd_lower = cmd.lower()
        for blocked in Config.BLOCKED_COMMANDS:
            if blocked and blocked.strip().lower() in cmd_lower:
                return False, f"Blocked: {blocked}"
        return True, ""


# =============================================================================
# TODO TRACKING
# =============================================================================

@dataclass
class TodoItem:
    id:          int
    description: str
    status:      str   # pending | in_progress | completed | blocked
    created:     str
    updated:     str
    notes:       str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "description": self.description,
                "status": self.status, "notes": self.notes}


_TODO_ICONS = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "blocked": "🚫"}


class TodoManager:
    def __init__(self, workspace: Path, session_id: str):
        self._file = workspace / ".lmagent" / "todos" / f"{session_id}.json"
        self.todos: List[TodoItem] = []
        self._next_id = 1
        self._load()

    def _load(self):
        if not self._file.exists(): return
        try:
            data          = json.loads(self._file.read_text(encoding="utf-8"))
            self.todos    = [TodoItem(**t) for t in data.get("todos", [])]
            self._next_id = data.get("next_id", 1)
        except Exception:
            self.todos, self._next_id = [], 1

    def _save(self):
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps({"todos": [vars(t) for t in self.todos],
                            "next_id": self._next_id}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            Log.error(f"Failed to save todos: {e}")

    def add(self, description: str, notes: str = "") -> Dict[str, Any]:
        now  = datetime.now().isoformat()
        item = TodoItem(id=self._next_id, description=description,
                        status="pending", created=now, updated=now, notes=notes)
        self.todos.append(item)
        self._next_id += 1
        self._save()
        return {"success": True, "todo_id": item.id, "description": description}

    def update_status(self, todo_id: int, new_status: str, notes: str = "") -> Dict[str, Any]:
        item = next((t for t in self.todos if t.id == todo_id), None)
        if not item:
            return {"success": False, "error": f"Todo #{todo_id} not found"}
        item.status  = new_status
        item.updated = datetime.now().isoformat()
        if notes: item.notes = notes
        self._save()
        return {"success": True, "todo_id": todo_id, "status": new_status}

    def complete(self, todo_id: int) -> Dict[str, Any]:
        return self.update_status(todo_id, "completed")

    def list_all(self) -> Dict[str, Any]:
        return {
            "success":   True,
            "todos":     [t.to_dict() for t in self.todos],
            "total":     len(self.todos),
            "completed": sum(1 for t in self.todos if t.status == "completed"),
        }

    def get_context(self) -> str:
        pending = [t for t in self.todos if t.status in ("pending", "in_progress")]
        if not pending: return ""
        lines = ["YOUR TODO LIST:"] + [
            f"{'🔄' if t.status == 'in_progress' else '⏳'} #{t.id}: {t.description}"
            for t in pending[:5]
        ]
        return "\n".join(lines)

    def display(self):
        if not self.todos:
            print(colored("  No todos yet", Colors.GRAY)); return
        print(colored("\n📋 Current Todo List:", Colors.BLUE, bold=True))
        print(colored("=" * 60, Colors.BLUE))
        for t in self.todos:
            print(f"{_TODO_ICONS.get(t.status, '❓')} #{t.id}: {t.description}")
            if t.notes:
                print(colored(f"   └─ {t.notes}", Colors.GRAY))


# =============================================================================
# PLAN MANAGEMENT
# =============================================================================

_PLAN_ICONS = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "skipped": "⏭️"}


class PlanManager:
    def __init__(self, workspace: Path, session_id: str):
        self._file = workspace / ".lmagent" / "plans" / f"{session_id}.json"
        self.plan: Optional[Dict[str, Any]] = None
        self.current_step_id: Optional[str] = None
        self._load()

    def _load(self):
        if not self._file.exists(): return
        try:
            data                 = json.loads(self._file.read_text(encoding="utf-8"))
            self.plan            = data.get("plan")
            self.current_step_id = data.get("current_step_id")
        except Exception:
            pass

    def _save(self):
        if not self.plan: return
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps({"plan": self.plan, "current_step_id": self.current_step_id},
                           indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            Log.error(f"Failed to save plan: {e}")

    def create(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        self.plan = {
            "title":   plan_data.get("title", "Execution Plan"),
            "created": datetime.now().isoformat(),
            "steps":   plan_data.get("steps", []),
            "status":  "active",
        }
        self.current_step_id = None
        self._save()
        return {"success": True}

    def _step_status(self, step_id: str) -> str:
        if not self.plan: return "unknown"
        return next((s["status"] for s in self.plan["steps"] if s["id"] == step_id), "unknown")

    def get_next_step(self) -> Optional[Dict[str, Any]]:
        if not self.plan: return None
        return next(
            (s for s in self.plan["steps"]
             if s["status"] == "pending"
             and all(self._step_status(d) == "completed"
                     for d in s.get("dependencies", []))),
            None,
        )

    def _update_step(self, step_id: str, status: str):
        if not self.plan: return
        for step in self.plan["steps"]:
            if step["id"] == step_id:
                step["status"] = status
                if status == "in_progress":
                    self.current_step_id = step_id
                    Log.plan(f"Started step: {step_id}")
                elif status == "completed":
                    if self.current_step_id == step_id:
                        self.current_step_id = None
                    Log.plan(f"Completed step: {step_id}")
                self._save()
                return

    def start_step(self, step_id: str):    self._update_step(step_id, "in_progress")
    def complete_step(self, step_id: str): self._update_step(step_id, "completed")

    def is_complete(self) -> bool:
        return bool(self.plan) and all(
            s["status"] not in ("pending", "in_progress") for s in self.plan["steps"]
        )

    def get_context(self) -> str:
        if not self.plan: return ""
        lines = [f"ACTIVE PLAN: {self.plan['title']}\n\nSTEPS:"]
        for i, s in enumerate(self.plan["steps"], 1):
            lines.append(
                f"{i}. {_PLAN_ICONS.get(s['status'], '❓')} {s['id']}: {s['description']}"
            )
            if s.get("verification"):
                lines.append(f"   Verify: {s['verification']}")
        nxt = self.get_next_step()
        if nxt:
            lines += [f"\nNEXT STEP: {nxt['id']} - {nxt['description']}",
                      "You MUST work on this step. When done, verify and mark complete."]
        return "\n".join(lines)

    def display(self):
        if not self.plan:
            print(colored("  No plan active", Colors.GRAY)); return
        print(colored(f"\n📋 Plan: {self.plan['title']}", Colors.BLUE, bold=True))
        print(colored("=" * 60, Colors.BLUE))
        for s in self.plan["steps"]:
            print(f"{_PLAN_ICONS.get(s['status'], '❓')} {s['id']}: {s['description']}")
            if s.get("verification"):
                print(colored(f"   Verify: {s['verification']}", Colors.GRAY))


# =============================================================================
# WAIT STATE
# =============================================================================

@dataclass
class WaitState:
    """Scheduled pause. LLM emits:  WAIT: <ISO_datetime>: <reason>"""
    reason:            str
    resume_after:      str
    context_on_resume: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]: return vars(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WaitState":
        return WaitState(**{k: d[k] for k in WaitState.__dataclass_fields__ if k in d})

    def is_ready(self) -> bool:
        try:
            return datetime.now() >= datetime.fromisoformat(self.resume_after)
        except ValueError:
            return True  # malformed → resume immediately


_WAIT_RE = re.compile(r'WAIT:\s*(\S+):\s*(.+)')


def detect_wait(content: str) -> Optional[WaitState]:
    """Parse WAIT protocol from LLM output."""
    if not content or "WAIT:" not in content: return None
    m = _WAIT_RE.search(content)
    if not m: return None
    resume_after = m.group(1).strip().rstrip(":")
    reason       = m.group(2).strip()
    try:
        datetime.fromisoformat(resume_after)
    except ValueError:
        Log.warning(f"WAIT token found but datetime invalid: '{resume_after}' — ignoring")
        return None
    return WaitState(
        reason=reason,
        resume_after=resume_after,
        context_on_resume=(
            f"Resuming from scheduled wait.\n"
            f"Wait reason: {reason}\n"
            f"Scheduled resume time: {resume_after}\n"
            f"Current time: {{current_time}}\n\n"
            f"Continue where you left off."
        ),
    )


# =============================================================================
# AGENT STATE & RESULT TYPES
# =============================================================================

@dataclass
class AgentState:
    session_id:          str
    iteration:           int
    loop_detector_state: Dict[str, Any]
    permission_mode:     str
    current_plan_step:   Optional[str]
    last_checkpoint:     str
    wait_state:          Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]: return vars(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentState":
        return AgentState(**{k: d.get(k) for k in AgentState.__dataclass_fields__})


class StateManager:
    def __init__(self, workspace: Path):
        self._dir = workspace / ".lmagent" / "state"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}_state.json"

    def save(self, session_id: str, state: AgentState):
        try:
            _atomic_write(self._path(session_id), state.to_dict())
        except Exception as e:
            Log.error(f"Failed to save state: {e}")

    def load(self, session_id: str) -> Optional[AgentState]:
        p = self._path(session_id)
        if not p.exists(): return None
        try:
            return AgentState.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            Log.error(f"Failed to load state: {e}")
            return None


@dataclass
class AgentEvent:
    type:      str
    data:      Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResult:
    status:        str
    final_answer:  str
    events:        List[AgentEvent]
    session_id:    str
    iterations:    int
    session_state: Optional[Dict[str, Any]] = None
    wait_until:    Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status":        self.status,
            "final_answer":  self.final_answer,
            "session_id":    self.session_id,
            "iterations":    self.iterations,
            "events":        [{"type": e.type, "data": e.data, "timestamp": e.timestamp}
                              for e in self.events],
            "session_state": self.session_state,
            "wait_until":    self.wait_until,
        }


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionManager:
    def __init__(self, workspace: Path):
        self._dir = workspace / ".lmagent" / "sessions"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def create(self, task: str, parent_session: Optional[str] = None) -> str:
        # uuid4 suffix prevents collisions when two sessions start in the same second.
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        now = datetime.now().isoformat()
        self.save(sid, [], {
            "id": sid, "task": task[:200], "created": now, "updated": now,
            "iterations": 0, "status": "active", "parent_session": parent_session,
        })
        return sid

    def save(self, sid: str, messages: List[Dict], meta: Dict):
        meta["updated"] = datetime.now().isoformat()
        try:
            _atomic_write(self._path(sid), {"metadata": meta, "messages": messages})
        except Exception as e:
            Log.error(f"Failed to save session: {e}")

    def load(self, sid: str) -> Optional[Tuple[List[Dict], Dict]]:
        p = self._path(sid)
        if not p.exists(): return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data["messages"], data["metadata"]
        except Exception as e:
            Log.error(f"Failed to load session: {e}")
            return None

    def list_recent(self, limit: int = 10) -> List[Dict]:
        sessions = []
        for f in sorted(self._dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                m = json.loads(f.read_text(encoding="utf-8"))["metadata"]
                sessions.append({
                    "id":         m["id"],
                    "task":       m.get("task", "")[:50],
                    "created":    m.get("created", ""),
                    "status":     m.get("status", "unknown"),
                    "iterations": m.get("iterations", 0),
                    "parent":     m.get("parent_session"),
                })
            except Exception:
                continue
        return sessions


# =============================================================================
# PROJECT & SOUL CONFIG
# =============================================================================

class ProjectConfig:
    @staticmethod
    def load(workspace: Path) -> str:
        cfg = workspace / ".lmagent.md"
        if cfg.exists():
            try:
                return cfg.read_text(encoding="utf-8")
            except Exception as e:
                Log.warning(f"Failed to load project config: {e}")
        return ""


class SoulConfig:
    DEFAULT = (
        "I am a focused, reliable digital coworker. "
        "I act before I narrate. I verify before I claim. "
        "I'm direct but friendly — brevity over verbosity."
    )

    @staticmethod
    def load(workspace: Path) -> str:
        soul_file = workspace / ".soul.md"
        if soul_file.exists():
            try:
                content = soul_file.read_text(encoding="utf-8").strip()
                if content:
                    Log.info("Soul loaded from .soul.md")
                    return content
            except Exception as e:
                Log.warning(f"Could not read .soul.md: {e}")
        return SoulConfig.DEFAULT


# =============================================================================
# MESSAGE SUMMARIZATION
# =============================================================================

class MessageSummarizer:
    @staticmethod
    def _simple(messages: List[Dict[str, Any]]) -> str:
        tools_used, files_modified, errors = [], [], 0
        for msg in messages:
            if msg.get("role") != "tool": continue
            try:
                result = json.loads(msg.get("content", "{}"))
                if result.get("success"):
                    tools_used.append(msg.get("name", "unknown"))
                    if msg.get("name") in ("write", "edit") and "path" in result:
                        files_modified.append(result["path"])
                else:
                    errors += 1
            except Exception:
                pass
        parts = []
        if tools_used:     parts.append(f"Tools: {dict(Counter(tools_used).most_common(3))}")
        if files_modified: parts.append(f"Modified: {', '.join(set(files_modified[:3]))}")
        if errors:         parts.append(f"Errors: {errors}")
        return " | ".join(parts) or "No significant activity"

    @staticmethod
    def _llm(messages: List[Dict[str, Any]]) -> Optional[str]:
        condensed = []
        for msg in messages[-15:]:
            role = msg.get("role", "")
            if role == "user":
                condensed.append(f"USER: {msg.get('content', '')[:150]}")
            elif role == "assistant" and msg.get("content"):
                condensed.append(f"AGENT: {msg['content'][:150]}")
            elif role == "tool":
                try:
                    r = json.loads(msg.get("content", "{}"))
                    condensed.append(f"{'✓' if r.get('success') else '✗'} {msg.get('name')}")
                except Exception:
                    pass
        try:
            resp = requests.post(
                Config.LLM_URL,
                json={
                    "model":    Config.LLM_MODEL or "default",
                    "messages": [{"role": "user", "content":
                                  f"Summarize this conversation in 2 sentences:\n"
                                  f"{chr(10).join(condensed)}\n\nSUMMARY:"}],
                    "temperature": 0.3, "max_tokens": 150,
                },
                headers={"Authorization": f"Bearer {Config.LLM_API_KEY}",
                         "Content-Type": "application/json"},
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    @staticmethod
    def summarize(messages: List[Dict[str, Any]]) -> str:
        if not Config.ENABLE_SUMMARIZATION:
            return MessageSummarizer._simple(messages)
        return MessageSummarizer._llm(messages) or MessageSummarizer._simple(messages)


# =============================================================================
# TASK STATE
# =============================================================================

@dataclass
class TaskState:
    objective:            str
    completion_gate:      str
    inventory_hash:       str
    total_count:          int
    processed_count:      int
    remaining_queue:      List[str]
    rename_map:           Dict[str, str]
    last_error:           str
    recovery_instruction: str
    next_action:          str
    last_updated:         str

    def to_dict(self) -> Dict[str, Any]: return vars(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskState":
        return TaskState(
            objective=d.get("objective", ""),
            completion_gate=d.get("completion_gate", ""),
            inventory_hash=d.get("inventory_hash", ""),
            total_count=d.get("total_count", 0),
            processed_count=d.get("processed_count", 0),
            remaining_queue=d.get("remaining_queue", []),
            rename_map=d.get("rename_map", {}),
            last_error=d.get("last_error", ""),
            recovery_instruction=d.get("recovery_instruction", ""),
            next_action=d.get("next_action", ""),
            last_updated=d.get("last_updated", ""),
        )

    def to_message(self) -> Dict[str, Any]:
        queue_preview  = ("\n".join(self.remaining_queue[:10])
                          + ("\n..." if len(self.remaining_queue) > 10 else ""))
        rename_preview = ("\n".join(f"  {k} -> {v}" for k, v in list(self.rename_map.items())[:5])
                          + ("\n..." if len(self.rename_map) > 5 else ""))
        return {"role": "system", "content": (
            f"[TASK STATE - DO NOT SUMMARIZE]\n"
            f"OBJECTIVE: {self.objective}\n"
            f"COMPLETION_GATE: {self.completion_gate}\n"
            f"PROGRESS: {self.processed_count}/{self.total_count} processed\n"
            f"INVENTORY_HASH: {self.inventory_hash}\n\n"
            f"REMAINING_QUEUE ({len(self.remaining_queue)} files):\n{queue_preview}\n\n"
            f"RENAME_MAP ({len(self.rename_map)} entries):\n{rename_preview}\n\n"
            f"NEXT_ACTION: {self.next_action}\n"
            f"LAST_ERROR: {self.last_error or 'None'}\n"
            f"[END TASK STATE]"
        )}

    @staticmethod
    def compute_inventory_hash(items: List[str]) -> str:
        return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()[:16]


class TaskStateManager:
    def __init__(self, workspace: Path, session_id: str):
        self._file = workspace / ".lmagent" / "state" / f"{session_id}_task.json"
        self.current_state: Optional[TaskState] = None

    def checkpoint(self, state: TaskState):
        state.last_updated = datetime.now().isoformat()
        self.current_state = state
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._file, state.to_dict())
        except Exception as e:
            Log.error(f"Failed to checkpoint task state: {e}")

    def load(self) -> Optional[TaskState]:
        if not self._file.exists(): return None
        try:
            self.current_state = TaskState.from_dict(
                json.loads(self._file.read_text(encoding="utf-8"))
            )
            return self.current_state
        except Exception as e:
            Log.error(f"Failed to load task state: {e}")
            return None

    def clear(self):
        self.current_state = None
        self._file.unlink(missing_ok=True)


# =============================================================================
# MESSAGE COMPACTION HELPERS
# =============================================================================

def _extract_task_state_msg(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next(
        (m for m in reversed(messages)
         if m.get("role") == "system"
         and "[TASK STATE - DO NOT SUMMARIZE]" in m.get("content", "")),
        None,
    )


def _build_progress_summary(messages: List[Dict[str, Any]]) -> str:
    """Build a human-readable summary of work done in the compacted messages."""
    task_state_msg = _extract_task_state_msg(messages)
    if task_state_msg:
        content       = task_state_msg.get("content", "")
        summary_lines = []
        for line in content.split("\n"):
            for key in ("OBJECTIVE:", "PROGRESS:", "COMPLETION_GATE:",
                        "NEXT_ACTION:", "LAST_ERROR:"):
                if line.startswith(key):
                    summary_lines.append(line)
                    break
        if summary_lines:
            return "PROGRESS SUMMARY (from task checkpoint):\n" + "\n".join(summary_lines)

    # Fallback: count actual successful tool results from the message list
    files_written: List[str] = []
    files_edited:  List[str] = []
    ps_commands = 0
    errors      = 0

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            result = json.loads(msg.get("content", "{}"))
            name   = msg.get("name", "")
            if result.get("success"):
                if name == "write":
                    if result.get("path"):
                        files_written.append(result["path"])
                elif name == "edit":
                    if result.get("path"):
                        files_edited.append(result["path"])
                elif name in ("shell", "powershell"):
                    ps_commands += 1
            else:
                errors += 1
        except Exception:
            pass

    if not files_written and not files_edited and not ps_commands:
        return "No measurable progress in summarized messages."

    parts: List[str] = []
    if files_written:
        unique = list(dict.fromkeys(files_written))
        tail   = f" +{len(unique) - 8} more" if len(unique) > 8 else ""
        parts.append(f"Files written ({len(unique)}): {', '.join(unique[:8])}{tail}")
    if files_edited:
        unique = list(dict.fromkeys(files_edited))
        tail   = f" +{len(unique) - 8} more" if len(unique) > 8 else ""
        parts.append(f"Files edited ({len(unique)}): {', '.join(unique[:8])}{tail}")
    if ps_commands:
        parts.append(f"Shell commands executed: {ps_commands}")
    if errors:
        parts.append(f"Errors encountered: {errors}")
    return "PROGRESS SUMMARY:\n" + "\n".join(parts)


def _extract_critical_work_msgs(messages: List[Dict[str, Any]],
                                keep: int = 20) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any], int]] = []
    for i, msg in enumerate(messages):
        content = str(msg.get("content", ""))
        role    = msg.get("role", "")
        if role == "tool" and msg.get("name") in ("read", "write", "edit", "shell", "powershell"):
            try:
                result = json.loads(content)
                if result.get("success"):
                    priority = (900 if msg.get("name") in ("shell", "powershell")
                                and any(k in result.get("stdout", "").lower()
                                        for k in ("rename", "move")) else 500)
                    scored.append((i, msg, priority)); continue
            except Exception:
                pass
        if role == "assistant":
            p = 300 if msg.get("tool_calls") else (100 if len(content) > 50 else 0)
            if p: scored.append((i, msg, p))
    scored.sort(key=lambda x: (x[2], x[0]), reverse=True)
    selected = sorted(idx for idx, _, _ in scored[:keep])
    return [messages[i] for i in selected]


def compact_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total_tokens = TokenCounter.count_messages_tokens(messages)

    if (total_tokens < Config.SUMMARIZATION_THRESHOLD
            or len(messages) <= Config.KEEP_RECENT_MESSAGES + 10):
        return messages

    Log.info(f"Compacting context: {total_tokens} tokens")
    system_msg     = messages[0]
    task_state_msg = _extract_task_state_msg(messages)

    last_plan_msg:  Optional[Dict[str, Any]] = None
    last_todo_msg:  Optional[Dict[str, Any]] = None
    reconcile_msgs: List[Dict[str, Any]]     = []
    regular_msgs:   List[Dict[str, Any]]     = []

    for msg in messages[1:]:
        if (msg.get("role") == "system"
                and "[TASK STATE - DO NOT SUMMARIZE]" in msg.get("content", "")):
            continue
        content = str(msg.get("content", ""))
        if "ACTIVE PLAN:" in content or "NEXT STEP:" in content:
            last_plan_msg = msg
        elif "YOUR TODO LIST:" in content:
            last_todo_msg = msg
        elif "[RECONCILE REQUIRED]" in content:
            reconcile_msgs.append(msg)
        else:
            regular_msgs.append(msg)

    # ── score every message by importance ─────────────────────────────────────
    # Build index → score map rather than using list.index() (which is O(n) and
    # breaks on duplicate object references).
    scored_indices: List[Tuple[int, int]] = []   # (index, priority)
    for i, msg in enumerate(regular_msgs):
        content = str(msg.get("content", ""))
        role    = msg.get("role", "")
        priority = 0

        if role == "tool":
            name = msg.get("name", "")
            try:
                result = json.loads(content)
                if result.get("success"):
                    if name in ("shell", "powershell"):
                        stdout = result.get("stdout", "").lower()
                        # Rename/move ops are high-value context
                        priority = 900 if any(k in stdout for k in
                                              ("rename", "move", "ren ", "mv ")) else 500
                    elif name in ("write", "edit"):
                        priority = 600
                    elif name in ("read",):
                        priority = 400
                    elif name in ("todo_complete", "plan_complete_step"):
                        priority = 700   # completion signals are high-value
                    elif name in ("task_state_update",):
                        priority = 650
                    else:
                        priority = 300
                else:
                    priority = 200   # failed results still useful for loop detection
            except Exception:
                priority = 100

        elif role == "assistant":
            if msg.get("tool_calls"):
                priority = 350
            elif len(content) > 50:
                priority = 150

        if priority > 0:
            scored_indices.append((i, priority))

    # Keep top-scored + always keep the tail half (minimum 25 msgs)
    scored_indices.sort(key=lambda x: (x[1], x[0]), reverse=True)
    keep_indices: set = {idx for idx, _ in
                         scored_indices[:Config.KEEP_RECENT_MESSAGES]}
    tail_start = max(0, len(regular_msgs) - max(25, len(regular_msgs) // 2))
    keep_indices |= set(range(tail_start, len(regular_msgs)))

    # Preserve recent read call+result pairs to avoid re-read loops
    for i in range(max(0, len(regular_msgs) - 20), len(regular_msgs)):
        msg = regular_msgs[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") == "read":
                    try:
                        path = json.loads(
                            tc["function"].get("arguments", "{}")
                        ).get("path", "")
                        if path:
                            keep_indices.add(i)
                            for j in range(i + 1, min(i + 3, len(regular_msgs))):
                                if (regular_msgs[j].get("role") == "tool"
                                        and regular_msgs[j].get("name") == "read"):
                                    keep_indices.add(j)
                                    break
                    except Exception:
                        pass

    # ── keep tool-call / result pairs atomic ───────────────────────────────────
    # BUG FIX: original only checked tool_calls[0]; an assistant turn can make
    # multiple calls and the matching one may be at any index.
    for i in sorted(list(keep_indices)):   # iterate a snapshot — set mutates
        if i >= len(regular_msgs):
            continue
        msg = regular_msgs[i]

        if msg.get("role") == "tool":
            # Find the assistant turn that owns this result
            tid = msg.get("tool_call_id")
            for j in range(i - 1, -1, -1):
                prev = regular_msgs[j]
                if prev.get("role") != "assistant":
                    continue
                # Check ALL tool calls in the assistant turn, not just [0]
                if any(tc.get("id") == tid
                       for tc in (prev.get("tool_calls") or [])):
                    keep_indices.add(j)
                    break

        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Keep ALL result messages for this assistant turn's tool calls
            call_ids = {tc.get("id") for tc in msg["tool_calls"]}
            for j in range(i + 1, len(regular_msgs)):
                r = regular_msgs[j]
                if (r.get("role") == "tool"
                        and r.get("tool_call_id") in call_ids):
                    keep_indices.add(j)
                    call_ids.discard(r.get("tool_call_id"))
                    if not call_ids:
                        break

    kept      = [regular_msgs[i] for i in sorted(keep_indices)]
    discarded = [regular_msgs[i] for i in range(len(regular_msgs))
                 if i not in keep_indices]

    result = [system_msg]
    if task_state_msg: result.append(task_state_msg)
    if last_plan_msg:  result.append(last_plan_msg)
    if last_todo_msg:  result.append(last_todo_msg)

    if discarded:
        result.append({"role": "system", "content": (
            f"[COMPACTION SUMMARY]\n"
            f"Summarized {len(discarded)} older messages to preserve context space.\n\n"
            f"{_build_progress_summary(discarded)}\n\n"
            f"IMPORTANT: The filesystem is now the source of truth. "
            f"Any file paths in the summary may be stale.\n"
            f"[END COMPACTION SUMMARY]"
        )})

    result.extend(kept)
    result.extend(reconcile_msgs)

    if task_state_msg and discarded:
        result.append({"role": "system", "content": (
            "[RECONCILIATION CHECKPOINT]\n\n"
            "Context was compacted. Before continuing:\n"
            "1. Review TASK STATE above for your objective and progress\n"
            "2. The filesystem is SOURCE OF TRUTH — ignore stale paths\n"
            "3. Use ls/glob to see current reality\n"
            "4. Continue with next file in REMAINING_QUEUE\n"
            "DO NOT repeat work. Check TASK STATE.processed_count.\n"
            "[END RECONCILIATION CHECKPOINT]"
        )})

    new_tokens = TokenCounter.count_messages_tokens(result)
    Log.success(f"Compacted: {total_tokens} → {new_tokens} tokens "
                f"({len(messages)} → {len(result)} msgs)")

    # ── safety valve: if still over threshold, drop more aggressively ─────────
    # This prevents the agent from hitting an infinite compaction loop when the
    # kept + injected system messages alone exceed the threshold.
    if new_tokens >= total_tokens * 0.95:
        Log.warning("Compaction barely reduced tokens — forcing aggressive trim")
        # Keep only system frames + last KEEP_RECENT_MESSAGES regular messages
        hard_kept = kept[-max(10, Config.KEEP_RECENT_MESSAGES // 2):]
        result = [system_msg]
        if task_state_msg: result.append(task_state_msg)
        if last_plan_msg:  result.append(last_plan_msg)
        result.append({"role": "system", "content": (
            "[AGGRESSIVE COMPACTION — context was very large]\n"
            f"{_build_progress_summary(discarded + kept[:-len(hard_kept)])}\n"
            "[END COMPACTION SUMMARY]"
        )})
        result.extend(hard_kept)
        result.extend(reconcile_msgs[-3:])  # keep only last 3 reconcile msgs
        new_tokens = TokenCounter.count_messages_tokens(result)
        Log.warning(f"Aggressive compaction: now {new_tokens} tokens, {len(result)} msgs")

    return result


# =============================================================================
# LOOP DETECTION
# =============================================================================

_PROGRESS_TOOLS = frozenset({
    "write", "edit", "mkdir", "shell", "powershell",
    "git_add", "git_commit", "git_branch", "task",
})


@dataclass
class LoopDetector:
    tool_history:       List[Tuple[str, str]] = field(default_factory=list)
    error_streak:       int = 0
    last_progress_iter: int = 0
    empty_iterations:   int = 0

    def track_tool(self, tool: str, args: Dict[str, Any], success: bool):
        self.tool_history.append((tool, json.dumps(args, sort_keys=True)[:50]))
        self.tool_history = self.tool_history[-30:]
        if success and tool in _PROGRESS_TOOLS:
            self.empty_iterations = 0

    def track_success(self, iteration: int):
        self.last_progress_iter = iteration
        self.error_streak       = 0
        self.empty_iterations   = 0

    def track_error(self): self.error_streak += 1
    def track_empty(self): self.empty_iterations += 1

    def check(self, iteration: int) -> Optional[str]:
        if self.empty_iterations >= Config.MAX_EMPTY_ITERATIONS:
            return f"Agent stopped responding ({self.empty_iterations} empty iterations)"
        if len(self.tool_history) >= Config.MAX_SAME_TOOL_STREAK:
            tool, count = Counter(
                t[0] for t in self.tool_history[-Config.MAX_SAME_TOOL_STREAK:]
            ).most_common(1)[0]
            if count >= Config.MAX_SAME_TOOL_STREAK:
                return f"Loop detected: '{tool}' repeated {count} times"
        since = iteration - self.last_progress_iter
        if since >= Config.MAX_NO_PROGRESS_ITERS:
            return f"No progress in {since} iterations"
        if self.error_streak >= Config.MAX_ERRORS:
            return f"{self.error_streak} consecutive errors"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_history":       self.tool_history,
                "error_streak":       self.error_streak,
                "last_progress_iter": self.last_progress_iter,
                "empty_iterations":   self.empty_iterations}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LoopDetector":
        ld = LoopDetector()
        ld.tool_history       = [tuple(x) for x in d.get("tool_history", [])]
        ld.error_streak       = d.get("error_streak", 0)
        ld.last_progress_iter = d.get("last_progress_iter", 0)
        ld.empty_iterations   = d.get("empty_iterations", 0)
        return ld


# =============================================================================
# MCP INTEGRATION
# =============================================================================

class MCPClient:
    """JSON-RPC 2.0 MCP server client with auto-restart and health checks."""

    def __init__(self, name: str, command: str, args: List[str], env: Dict[str, str]):
        self.name    = name
        self.command = command
        self.args    = args
        self.env     = env
        self.process: Optional[subprocess.Popen] = None
        self.tools:   List[Dict[str, Any]]        = []
        self._request_id  = 0
        self.healthy      = False
        self._pending:    Dict[int, threading.Event] = {}
        self._responses:  Dict[int, Dict[str, Any]]  = {}
        self._reader:     Optional[threading.Thread]  = None
        self._lock        = threading.Lock()
        self._shutdown    = threading.Event()
        self._last_stderr = ""

    def start(self):
        self._force_kill()
        try:
            self._shutdown.clear()
            popen_kwargs: Dict[str, Any] = dict(
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **self.env},
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if _IS_WINDOWS:
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            self.process = subprocess.Popen(
                [self.command] + self.args, **popen_kwargs
            )
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            time.sleep(0.05)
            self._send("initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "lm-agent", "version": VERSION},
            }, timeout=15)
            result     = self._send("tools/list", {}, timeout=5)
            self.tools = (result or {}).get("tools", [])
            self.healthy = True
        except Exception as e:
            Log.error(f"Failed to start MCP '{self.name}': {e}")
            if self._last_stderr:
                Log.error(f"MCP '{self.name}' stderr: {self._last_stderr[:200]}")
            self._force_kill()

    def close(self): self._force_kill()

    def _force_kill(self):
        self._shutdown.set()
        if self.process:
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
        with self._lock:
            self._pending.clear()
            self._responses.clear()
            self.healthy = False
        self._reader = None
        self._shutdown.clear()

    def _read_stderr(self):
        if not self.process: return
        try:
            while not self._shutdown.is_set():
                if not self.process or self.process.poll() is not None: break
                line = self.process.stderr.readline()
                if line: self._last_stderr = (self._last_stderr + line)[-1000:]
        except Exception:
            pass

    def _read_loop(self):
        while not self._shutdown.is_set():
            if not self.process or self.process.poll() is not None: break
            try:
                line = self.process.stdout.readline()
                if not line: break
                line = line.strip()
                if not line: continue
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = resp.get("id")
                if rid is not None:
                    with self._lock:
                        self._responses[rid] = resp
                        if rid in self._pending:
                            self._pending[rid].set()
            except Exception as e:
                if not self._shutdown.is_set():
                    Log.error(f"MCP '{self.name}' reader: {e}")
                break
        with self._lock:
            self.healthy = False
            for ev in self._pending.values(): ev.set()

    def _send(self, method: str, params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        if not self.process: raise RuntimeError("MCP server not running")
        if not self._reader or not self._reader.is_alive():
            raise RuntimeError("MCP reader thread not running")
        self._request_id += 1
        rid   = self._request_id
        event = threading.Event()
        with self._lock:
            self._pending[rid] = event
        try:
            self.process.stdin.write(
                json.dumps({"jsonrpc": "2.0", "id": rid,
                            "method": method, "params": params}) + "\n"
            )
            self.process.stdin.flush()
        except Exception as e:
            with self._lock: self._pending.pop(rid, None)
            raise RuntimeError(f"Failed to send request: {e}")
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
                self._responses.pop(rid, None)
            raise TimeoutError(f"Timeout after {timeout}s (method={method})")
        with self._lock:
            resp = self._responses.pop(rid, None)
            self._pending.pop(rid, None)
        if resp is None: raise RuntimeError("No response (server crashed)")
        if "error" in resp: raise RuntimeError(resp["error"].get("message", "Unknown error"))
        return resp.get("result", {})

    def health_check(self) -> bool:
        if not self.process or self.process.poll() is not None: self.healthy = False
        if not self._reader or not self._reader.is_alive(): self.healthy = False
        return self.healthy

    def call_tool(self, tool_name: str, arguments: Dict[str, Any],
                  max_attempts: int = 3) -> Dict[str, Any]:
        for attempt in range(max_attempts):
            if not self.health_check():
                Log.warning(f"MCP '{self.name}' unhealthy — restarting "
                             f"(attempt {attempt + 1}/{max_attempts})")
                self._force_kill()
                self.start()
                if not self.healthy:
                    if attempt == max_attempts - 1:
                        return {"success": False,
                                "error": f"MCP '{self.name}' failed after {max_attempts} attempts"}
                    time.sleep(0.2); continue
            try:
                result = self._send("tools/call",
                                    {"name": tool_name, "arguments": arguments}, timeout=10)
                return {"success": True, "result": result}
            except Exception as e:
                Log.warning(f"MCP '{self.name}' call failed "
                             f"(attempt {attempt + 1}): {str(e)[:100]}")
                self._force_kill()
                if attempt < max_attempts - 1:
                    self.start(); time.sleep(0.1)
                else:
                    return {"success": False,
                            "error": f"MCP call failed after {max_attempts} attempts: {e}"}
        return {"success": False, "error": "Unexpected error in call_tool"}


class MCPManager:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.clients:  List[MCPClient] = []

    def load_servers(self):
        if not Config.ENABLE_MCP: return
        cfg_file = self.workspace / ".lmagent" / "mcp.json"
        if not cfg_file.exists(): return
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            for name, settings in cfg.get("mcpServers", {}).items():
                command = settings.get("command", "")
                if not command: continue
                client = MCPClient(name, command,
                                   settings.get("args", []), settings.get("env", {}))
                client.start()
                if client.process and client.healthy:
                    self.clients.append(client)
                    Log.info(f"MCP '{name}' started")
                else:
                    Log.warning(f"MCP '{name}' failed to start")
            if self.clients:
                Log.info(f"Loaded {len(self.clients)} MCP servers")
        except Exception as e:
            Log.warning(f"Failed to load MCP: {e}")

    def get_all_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for client in self.clients:
            if not client.health_check(): continue
            for tool in client.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name":        f"mcp_{client.name}_{tool['name']}",
                        "description": tool.get("description", f"MCP: {tool['name']}"),
                        "parameters":  tool.get("inputSchema",
                                                {"type": "object", "properties": {}}),
                    },
                    "_mcp_client": client.name,
                    "_mcp_tool":   tool["name"],
                })
        return tools

    def call_tool(self, full_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not full_name.startswith("mcp_"):
            return {"success": False, "error": "Not an MCP tool"}
        remainder = full_name[4:]
        for client in self.clients:
            if remainder.startswith(client.name + "_"):
                return client.call_tool(remainder[len(client.name) + 1:], arguments)
        return {"success": False,
                "error": f"No matching MCP client for '{full_name}'. "
                         f"Available: {[c.name for c in self.clients]}"}

    def get_status(self) -> Dict[str, bool]:
        return {c.name: c.health_check() for c in self.clients}

    def close_all(self):
        for c in self.clients:
            try: c.close()
            except Exception: pass
        self.clients.clear()
