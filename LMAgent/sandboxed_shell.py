#!/usr/bin/env python3
"""
sandboxed_shell.py — Cross-platform sandboxed subprocess execution.

ONE container, used forever
----------------------------
A single hardened Docker container is created on first use and given a
fixed name (SANDBOX_CONTAINER_NAME). Every call to run_sandboxed() finds
and reuses it by name — no CID file, no fragile state.

A new container is only made when the old one is GENUINELY gone:
  • First ever call (nothing exists)
  • The container was manually deleted from Docker Desktop
  • The container crashed / OOM-killed by the OS

Cleanup order
-------------
1. atexit    — normal exit / /new / quit
2. SIGTERM   — killed by task manager or `docker stop`
3. SIGINT    — Ctrl+C
4. Watchdog  — daemon thread catches interpreter teardown after hard crashes
(SIGKILL cannot be caught, but atexit + watchdog covers all real scenarios.)

Platform behaviour
------------------
  Windows / FORCE_DOCKER=True  → persistent hardened Docker container
  macOS / Linux                → process group + rlimits (set FORCE_DOCKER=True for Docker)
  Docker unavailable           → process-group fallback + loud warning

Requirements
------------
    pip install psutil      # recommended
    pip install docker      # required for Docker backend
    Docker Desktop          # must be running (Windows / FORCE_DOCKER)
"""

from __future__ import annotations

import atexit
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DOCKER_IMAGE            = "python:3.12-slim"
DOCKER_NETWORK          = "bridge"         # "none" = air-gapped, "bridge" = outbound internet
FORCE_DOCKER            = False            # True = use Docker on macOS/Linux too
SANDBOX_CONTAINER_NAME  = "claude-sandbox" # fixed name — survives restarts without a CID file

DEFAULT_TIMEOUT_SECS  = 30
DEFAULT_MAX_OUTPUT    = 32_768   # bytes
DEFAULT_MAX_MEMORY_MB = 512

# ---------------------------------------------------------------------------
# Platform / optional imports
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"
_IS_POSIX   = not _IS_WINDOWS

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

if _IS_POSIX:
    import resource

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_lock               = threading.RLock()
_docker_client      = None
_docker_available   = None   # None = unchecked, True/False = known
_sandbox_container  = None   # the ONE container for this process
_cleanup_done       = False
_cleanup_registered = False
_fallback_warned    = False

# ---------------------------------------------------------------------------
# Terminal output (stderr only — LLM never sees these)
# ---------------------------------------------------------------------------

_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _info(msg: str) -> None:
    sys.stderr.write(f"[sandbox] {msg}\n")
    sys.stderr.flush()


def _warn_fallback_header() -> None:
    global _fallback_warned
    with _lock:
        if _fallback_warned:
            return
        _fallback_warned = True
    sys.stderr.write(
        f"\n{_BOLD}{_RED}"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  ⚠  SANDBOX WARNING — INSECURE FALLBACK MODE ACTIVE         ║\n"
        "║                                                              ║\n"
        "║  Docker is not running. Shell commands execute on your HOST  ║\n"
        "║  machine with NO filesystem isolation. The LLM can access   ║\n"
        "║  files outside the workspace.                               ║\n"
        "║  Start Docker Desktop to restore full isolation.             ║\n"
        f"╚══════════════════════════════════════════════════════════════╝{_RESET}\n\n"
    )
    sys.stderr.flush()


def _warn_fallback_command(cmd: str) -> None:
    short = cmd if len(cmd) <= 60 else cmd[:57] + "..."
    sys.stderr.write(
        f"{_YELLOW}{_BOLD}[INSECURE FALLBACK]{_RESET}"
        f"{_YELLOW} Running on HOST (no Docker): {short}{_RESET}\n"
    )
    sys.stderr.flush()


def _warn_docker_restored() -> None:
    global _fallback_warned
    with _lock:
        _fallback_warned = False
    sys.stderr.write(
        f"\n{_BOLD}{_GREEN}[SANDBOX] Docker available — secure isolation restored.{_RESET}\n"
    )
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Docker client
# ---------------------------------------------------------------------------





def _check_docker() -> bool:
    """Probe Docker daemon; update globals under lock. Returns True if live."""
    global _docker_available, _docker_client
    try:
        import docker
        client = docker.from_env()
        client.ping()
        with _lock:
            was_down          = _docker_available is False
            _docker_available = True
            _docker_client    = client
        if was_down:
            _warn_docker_restored()
        return True
    except Exception:
        with _lock:
            _docker_available = False
            _docker_client    = None
        return False


def _get_docker_client():
    """Return a live Docker client, or None."""
    with _lock:
        client = _docker_client if (_docker_available is True and _docker_client) else None
    if client is not None:
        try:
            client.ping()
            return client
        except Exception:
            pass
    return _docker_client if _check_docker() else None


def _ensure_image(client) -> None:
    try:
        client.images.get(DOCKER_IMAGE)
    except Exception:
        _info(f"Pulling {DOCKER_IMAGE!r} (first run only)…")
        client.images.pull(DOCKER_IMAGE)
        _info("Image ready.")


# ---------------------------------------------------------------------------
# Container singleton — looked up by fixed name, not a CID file
# ---------------------------------------------------------------------------

def _container_is_alive(container) -> bool:
    """Return True if the container object refers to a running container."""
    try:
        container.reload()
        return container.status == "running"
    except Exception:
        return False


def _find_existing_container(client):
    """
    Look up the sandbox container by its fixed name.
      Running → return immediately.
      Stopped → restart and return.
      Missing → return None (caller will create it).
    """
    try:
        container = client.containers.get(SANDBOX_CONTAINER_NAME)
    except Exception:
        return None  # name not found — caller will create it

    if _container_is_alive(container):
        _info(f"Reconnected to running container '{SANDBOX_CONTAINER_NAME}' ({container.short_id})")
        return container

    # Stopped/paused/exited — try to restart
    if container.status in ("exited", "created", "paused"):
        _info(f"Container '{SANDBOX_CONTAINER_NAME}' is stopped — restarting…")
        try:
            container.start()
            container.reload()
            if _container_is_alive(container):
                _info(f"Container '{SANDBOX_CONTAINER_NAME}' restarted ({container.short_id})")
                return container
        except Exception as e:
            _info(f"Failed to restart '{SANDBOX_CONTAINER_NAME}': {e} — will recreate")

    # Dead and unrecoverable — remove so a fresh one can take the name
    _info(f"Removing dead container '{SANDBOX_CONTAINER_NAME}' to recreate it…")
    try:
        container.remove(force=True)
    except Exception:
        pass
    return None


def _get_or_create_container(workspace: Path, max_memory_mb: int):
    """
    Return the one sandbox container for this process.

    Decision order:
      1. In-memory ref alive  → reuse it (fast path, no Docker API call)
      2. Named container      → reconnect or restart
      3. Nothing found        → create a new container with the fixed name
    """
    global _sandbox_container, _cleanup_done, _cleanup_registered

    with _lock:
        # ── 1. Fast path: in-memory reference ────────────────────────────────
        if _sandbox_container is not None:
            if _container_is_alive(_sandbox_container):
                return _sandbox_container
            _info(f"Container {_sandbox_container.short_id} is no longer running — will reconnect or recreate")
            _sandbox_container = None

        client = _get_docker_client()
        if client is None:
            return None

        _ensure_image(client)

        # ── 2. Find by fixed name ─────────────────────────────────────────────
        existing = _find_existing_container(client)
        if existing is not None:
            _sandbox_container = existing
            _cleanup_done = False
            _register_hooks_once()
            return _sandbox_container

        # ── 3. Nothing reusable — create a brand-new container ───────────────
        _info(f"Creating new sandbox container '{SANDBOX_CONTAINER_NAME}' ({DOCKER_IMAGE})…")

        workspace_abs = str(workspace.resolve())

        # NOTE: masked_paths is NOT a supported kwarg in docker-py's high-level
        # containers.run() or create_host_config(). However, Docker automatically
        # applies default masked paths (covering /proc/kcore, /proc/sys,
        # /proc/sysrq-trigger, etc.) for any non-privileged container when
        # MaskedPaths is unset in the HostConfig. We rely on that behaviour.
        container = client.containers.run(
            image=DOCKER_IMAGE,
            name=SANDBOX_CONTAINER_NAME,
            command=["sh", "-c", "while true; do sleep 3600; done"],
            volumes={workspace_abs: {"bind": "/workspace", "mode": "rw"}},
            tmpfs={"/tmp": "size=64m,mode=1777"},
            working_dir="/workspace",
            mem_limit=f"{max_memory_mb}m",
            memswap_limit=f"{max_memory_mb}m",
            cpu_period=100_000,
            cpu_quota=90_000,
            pids_limit=128,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            read_only=True,
            network_mode=DOCKER_NETWORK,
            remove=False,
            tty=False,
            detach=True,
            stdout=True,
            stderr=True,
        )

        _sandbox_container = container
        _cleanup_done = False
        _info(f"Sandbox container ready: '{SANDBOX_CONTAINER_NAME}' ({container.short_id})")

        _register_hooks_once()
        return container


# ---------------------------------------------------------------------------
# Cleanup — atexit, signals, watchdog
# ---------------------------------------------------------------------------

def _cleanup_container() -> None:
    """
    Stop (do NOT remove) the sandbox container on process exit.
    The next process finds it by name and restarts it.
    Safe to call multiple times — only the first call does work.
    """
    global _sandbox_container, _cleanup_done

    with _lock:
        if _cleanup_done:
            return
        _cleanup_done      = True
        container          = _sandbox_container
        _sandbox_container = None

    if container is None:
        return

    try:
        _info(f"Stopping '{SANDBOX_CONTAINER_NAME}' ({container.short_id}) — will reuse on next run")
        container.stop(timeout=5)
        _info("Container stopped.")
    except Exception:
        pass


def _register_hooks_once() -> None:
    """Register atexit, signal handlers, and watchdog — exactly once per process."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    atexit.register(_cleanup_container)

    def _handler(signum, frame):
        _cleanup_container()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass

    _start_watchdog()


def _start_watchdog() -> None:
    """
    Daemon thread that fires _cleanup_container() during interpreter teardown.
    Covers hard crashes where atexit cannot run.
    """
    parent_pid = os.getpid()

    def _watch():
        while True:
            time.sleep(5)
            try:
                if os.getpid() != parent_pid:
                    _cleanup_container()
                    return
            except Exception:
                _cleanup_container()
                return

    t = threading.Thread(target=_watch, daemon=True, name="sandbox-watchdog")
    t.start()


# ---------------------------------------------------------------------------
# Docker command execution
# ---------------------------------------------------------------------------

def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _run_docker(
    cmd: str,
    workspace: Path,
    timeout: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> Tuple[str, int]:
    """Fire cmd into the persistent sandbox container via exec_run()."""
    container = _get_or_create_container(workspace, max_memory_mb)
    if container is None:
        raise RuntimeError("Docker unavailable — cannot create sandbox container.")

    wrapped = f"timeout {timeout} sh -c {_sh_quote(cmd)}"

    try:
        result = container.exec_run(
            cmd=["sh", "-c", wrapped],
            workdir="/workspace",
            demux=False,
            tty=False,
            stdin=False,
        )

        exit_code = result.exit_code if result.exit_code is not None else -1
        raw       = result.output or b""
        truncated = len(raw) > max_output_bytes
        output    = raw[:max_output_bytes].decode("utf-8", errors="replace")
        if truncated:
            output += "\n[output truncated — 32 KB limit reached]"
        if exit_code == 124:
            output = f"[TIMEOUT after {timeout}s]\n" + output

        return output, exit_code

    except Exception as e:
        with _lock:
            global _sandbox_container
            if _sandbox_container is not None and not _container_is_alive(_sandbox_container):
                _info("Container died during exec_run — will reconnect or recreate on next call")
                _sandbox_container = None
            else:
                _info("exec_run failed but container is still alive — keeping it")
        raise RuntimeError(f"exec_run failed: {e}") from e


# ---------------------------------------------------------------------------
# POSIX process-group backend (fallback when Docker is unavailable)
# ---------------------------------------------------------------------------

def _posix_preexec(max_memory_mb: int) -> None:
    os.setsid()
    mem = max_memory_mb * 1024 * 1024
    for limit in (resource.RLIMIT_AS,):
        try:
            resource.setrlimit(limit, (mem, mem))
        except (ValueError, resource.error):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (ValueError, resource.error):
        pass


def _run_process_group(
    cmd: str,
    workspace: Path,
    timeout: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> Tuple[str, int]:
    popen_kwargs: dict = dict(
        shell=True,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if _IS_POSIX:
        popen_kwargs["preexec_fn"] = lambda: _posix_preexec(max_memory_mb)

    proc = subprocess.Popen(cmd, **popen_kwargs)

    chunks: list[bytes] = []
    total = 0
    trunc = False

    def _reader():
        nonlocal total, trunc
        assert proc.stdout
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            if total < max_output_bytes:
                room = max_output_bytes - total
                chunks.append(chunk[:room])
                total += min(len(chunk), room)
                if total >= max_output_bytes:
                    trunc = True

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_proc(proc.pid)
        reader.join(timeout=2)
        out = b"".join(chunks).decode("utf-8", errors="replace")
        return f"[TIMEOUT after {timeout}s]\n{out}", -1

    _sweep(proc.pid)
    reader.join(timeout=5)
    out = b"".join(chunks).decode("utf-8", errors="replace")
    if trunc:
        out += "\n[output truncated]"
    return out, proc.returncode


def _kill_proc(pid: int) -> None:
    if _IS_POSIX:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    _sweep(pid)


def _sweep(pid: int) -> None:
    if not _HAVE_PSUTIL:
        return
    try:
        for child in psutil.Process(pid).children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sandboxed(
    cmd: str,
    workspace: Path,
    timeout: int          = DEFAULT_TIMEOUT_SECS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT,
    max_memory_mb: int    = DEFAULT_MAX_MEMORY_MB,
) -> Tuple[str, int]:
    """
    Run *cmd* in the best available sandbox and return (output, exit_code).

    Windows / FORCE_DOCKER=True:
        Docker up   → ONE persistent hardened container, reused forever
        Docker down → process-group fallback with loud terminal warning
    macOS / Linux (default):
        → process group + rlimits  (set FORCE_DOCKER=True for Docker)
    """
    # Ensure the workspace directory exists before either backend tries to use
    # it. subprocess.Popen raises FileNotFoundError on a missing cwd, and
    # Docker's bind-mount also fails if the host path doesn't exist.
    workspace.mkdir(parents=True, exist_ok=True)

    want_docker = _IS_WINDOWS or FORCE_DOCKER

    if want_docker:
        if _get_docker_client() is not None:
            return _run_docker(cmd, workspace, timeout, max_output_bytes, max_memory_mb)
        _warn_fallback_header()
        _warn_fallback_command(cmd)

    return _run_process_group(cmd, workspace, timeout, max_output_bytes, max_memory_mb)


def sandbox_info() -> dict:
    """Return a snapshot of the current sandbox configuration."""
    want_docker = _IS_WINDOWS or FORCE_DOCKER
    docker_up   = _check_docker() if want_docker else False

    if want_docker and docker_up:
        with _lock:
            cid = _sandbox_container.short_id if _sandbox_container else "not started yet"
        backend = f"docker_hardened_persistent (name: {SANDBOX_CONTAINER_NAME}, id: {cid})"
    elif want_docker:
        backend = "process_group+rlimit (INSECURE FALLBACK — Docker not running)"
    else:
        backend = "process_group+rlimit"

    return {
        "platform":           platform.system(),
        "backend":            backend,
        "docker_available":   docker_up,
        "docker_image":       DOCKER_IMAGE if want_docker else None,
        "docker_network":     DOCKER_NETWORK if want_docker else None,
        "container_name":     SANDBOX_CONTAINER_NAME if want_docker else None,
        "docker_caps":        "ALL dropped" if (want_docker and docker_up) else None,
        "docker_root_fs":     "read-only"   if (want_docker and docker_up) else None,
        "force_docker":       FORCE_DOCKER,
        "psutil":             _HAVE_PSUTIL,
        "default_timeout":    DEFAULT_TIMEOUT_SECS,
        "default_memory_mb":  DEFAULT_MAX_MEMORY_MB,
    }
