#!/usr/bin/env python3
"""
sandboxed_shell.py — Cross-platform sandboxed subprocess execution.

Docker backend: EPHEMERAL containers
--------------------------------------
Each call to run_sandboxed() spins up a brand-new hardened Docker container,
runs exactly one command, captures its output, then removes the container in
a finally-block — even if the command crashes or times out.

There is NO persistent container, NO .sandbox_cid file, NO atexit / signal
plumbing needed for Docker: the container is always removed before the call
returns.

Hardening (same as before)
--------------------------
  - All Linux capabilities dropped
  - no-new-privileges secopt
  - Read-only root filesystem
  - /tmp and sensitive /proc paths on tmpfs
  - Memory + swap limit
  - CPU quota (90 % of one core)
  - PID limit (128)
  - Configurable network mode (default "none" = fully air-gapped)

Platform behaviour
------------------
  Windows / FORCE_DOCKER=True  → ephemeral hardened Docker container per call
  macOS / Linux                → process group + rlimits
  Docker unavailable           → process-group fallback + loud terminal warning

Requirements
------------
    pip install psutil      # recommended (child-process sweep)
    pip install docker      # required for Docker backend
    Docker Desktop          # must be running (Windows / FORCE_DOCKER)
"""

from __future__ import annotations

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

DOCKER_IMAGE   = "python:3.12-slim"
DOCKER_NETWORK = "none"    # "none" = fully air-gapped (safest), "bridge" = outbound internet
FORCE_DOCKER   = False     # True = use Docker on macOS / Linux too

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
# Module-level state — mutations go through _lock
# ---------------------------------------------------------------------------

_lock             = threading.Lock()
_docker_client    = None
_docker_available = None   # None = unchecked, True / False = known
_fallback_warned  = False
_image_pulled     = False  # pull once per process, not on every call

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
        "║  machine with NO filesystem isolation. The process can       ║\n"
        "║  access files outside the workspace.                        ║\n"
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
# Docker client management
# ---------------------------------------------------------------------------

# Sensitive /proc paths hidden inside containers via tmpfs overlays
_MASKED_PROC_PATHS = [
    "/proc/mounts",
    "/proc/net",
    "/proc/sys",
    "/proc/sched_debug",
    "/proc/timer_list",
]


def _check_docker() -> bool:
    """Probe Docker daemon; update globals under lock. Returns True if live."""
    global _docker_available, _docker_client
    try:
        import docker
        client = docker.from_env()
        client.ping()
        with _lock:
            was_down = _docker_available is False
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
            pass  # fall through and re-probe
    return _docker_client if _check_docker() else None


def _ensure_image(client) -> None:
    """Pull the image once per process if it is not already present locally."""
    global _image_pulled
    with _lock:
        if _image_pulled:
            return
    try:
        client.images.get(DOCKER_IMAGE)
        with _lock:
            _image_pulled = True
    except Exception:
        _info(f"Pulling {DOCKER_IMAGE!r} (first run only)…")
        client.images.pull(DOCKER_IMAGE)
        with _lock:
            _image_pulled = True
        _info("Image ready.")


# ---------------------------------------------------------------------------
# Docker command execution — EPHEMERAL (one container per command)
# ---------------------------------------------------------------------------

def _sh_quote(s: str) -> str:
    """Minimal POSIX shell quoting."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _run_docker(
    cmd: str,
    workspace: Path,
    timeout: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> Tuple[str, int]:
    """
    Run cmd in a brand-new ephemeral container.

    The container is always removed in the finally-block — whether the command
    succeeds, raises, or times out.  No persistent state is left behind.
    """
    client = _get_docker_client()
    if client is None:
        raise RuntimeError("Docker unavailable — cannot create sandbox container.")

    _ensure_image(client)

    workspace_abs = str(workspace.resolve())
    volumes = {workspace_abs: {"bind": "/workspace", "mode": "rw"}}
    tmpfs   = {"/tmp": "size=64m,mode=1777"}
    for p in _MASKED_PROC_PATHS:
        tmpfs[p] = "size=0,ro"

    # Wrap the user command in shell timeout so the process itself is bounded.
    wrapped = f"timeout {timeout} sh -c {_sh_quote(cmd)}"

    container = None
    try:
        # detach=True so we get a container object back and can call .wait()
        # with our own Python-level timeout.
        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=["sh", "-c", wrapped],
            volumes=volumes,
            tmpfs=tmpfs,
            working_dir="/workspace",
            # Resource limits
            mem_limit=f"{max_memory_mb}m",
            memswap_limit=f"{max_memory_mb}m",
            cpu_period=100_000,
            cpu_quota=90_000,       # 90 % of one core
            pids_limit=128,
            # Capability hardening
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            # Filesystem hardening
            read_only=True,
            # Network
            network_mode=DOCKER_NETWORK,
            # Lifecycle — do NOT auto-remove; we call remove() ourselves
            remove=False,
            tty=False,
            detach=True,
            stdout=True,
            stderr=True,
        )

        _info(f"Ephemeral sandbox started ({container.short_id})")

        # Wait for the container to finish, with a Python-level safety net.
        try:
            result = container.wait(timeout=timeout + 5)   # +5 s grace
        except Exception:
            # Timed out at Python level (shell timeout should have fired first)
            try:
                container.kill()
            except Exception:
                pass
            raw = container.logs(stdout=True, stderr=True) or b""
            output = raw[:max_output_bytes].decode("utf-8", errors="replace")
            return f"[TIMEOUT after {timeout}s]\n{output}", -1

        exit_code = result.get("StatusCode", -1)

        raw       = container.logs(stdout=True, stderr=True) or b""
        truncated = len(raw) > max_output_bytes
        output    = raw[:max_output_bytes].decode("utf-8", errors="replace")
        if truncated:
            output += "\n[output truncated — 32 KB limit reached]"
        if exit_code == 124:
            output = f"[TIMEOUT after {timeout}s]\n" + output

        return output, exit_code

    finally:
        # Always remove — this is the guarantee that no containers leak.
        if container is not None:
            try:
                container.remove(force=True)
                _info(f"Ephemeral sandbox removed ({container.short_id})")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# POSIX process-group backend (fallback when Docker is not available)
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
        Docker up   → ephemeral hardened container (auto-removed when done)
        Docker down → process-group fallback with loud terminal warning
    macOS / Linux (default):
        → process group + rlimits  (set FORCE_DOCKER=True for Docker)
    """
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
        backend = f"docker_hardened_ephemeral (image: {DOCKER_IMAGE})"
    elif want_docker:
        backend = "process_group+rlimit (INSECURE FALLBACK — Docker not running)"
    else:
        backend = "process_group+rlimit"

    return {
        "platform":          platform.system(),
        "backend":           backend,
        "docker_available":  docker_up,
        "docker_image":      DOCKER_IMAGE if want_docker else None,
        "docker_network":    DOCKER_NETWORK if want_docker else None,
        "docker_caps":       "ALL dropped" if (want_docker and docker_up) else None,
        "docker_root_fs":    "read-only"   if (want_docker and docker_up) else None,
        "docker_lifetime":   "ephemeral (removed after each command)" if (want_docker and docker_up) else None,
        "force_docker":      FORCE_DOCKER,
        "psutil":            _HAVE_PSUTIL,
        "default_timeout":   DEFAULT_TIMEOUT_SECS,
        "default_memory_mb": DEFAULT_MAX_MEMORY_MB,
    }
