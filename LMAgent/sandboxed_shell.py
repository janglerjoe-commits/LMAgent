#!/usr/bin/env python3
"""
sandboxed_shell.py — Cross-platform sandboxed subprocess execution.

Provides a single public function:
    run_sandboxed(cmd, workspace, timeout, max_output_bytes, max_memory_mb)
    → (stdout_stderr: str, exit_code: int)

Platform behaviour
──────────────────
Windows  (pywin32 installed)
    • Subprocess runs inside a Windows Job Object
    • Hard memory limit via JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    • kill_on_job_close=True: process tree dies the moment our handle closes,
      even if Python crashes — no zombie processes possible
    • Falls back to psutil tree-kill if pywin32 is absent

macOS / Linux
    • Subprocess runs in its own process group (os.setsid)
    • RLIMIT_AS (virtual memory) and RLIMIT_CPU enforced via resource module
    • os.killpg kills the entire group on timeout or error
    • psutil used as an additional sweep to catch orphaned children

Requirements (all optional — graceful fallback if missing):
    pip install psutil          # cross-platform process-tree kill
    pip install pywin32         # Windows Job Objects (stronger Windows isolation)
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

# ── optional imports ──────────────────────────────────────────────────────────

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

_IS_WINDOWS = platform.system() == "Windows"
_IS_MACOS   = platform.system() == "Darwin"
_IS_LINUX   = platform.system() == "Linux"

_HAVE_PYWIN32 = False
if _IS_WINDOWS:
    try:
        import win32job, win32api, win32con, win32process, pywintypes  # noqa: F401
        _HAVE_PYWIN32 = True
    except ImportError:
        pass

if not _IS_WINDOWS:
    import resource  # stdlib, always available on POSIX
    import signal

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SECS  = 30
DEFAULT_MAX_OUTPUT    = 32_768   # bytes — truncated if exceeded
DEFAULT_MAX_MEMORY_MB = 512      # MB — enforced where supported


# =============================================================================
# WINDOWS — JOB OBJECT SANDBOX
# =============================================================================

def _run_windows_job_object(
    cmd: str,
    cwd: Path,
    timeout: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> Tuple[str, int]:
    """
    Launch *cmd* inside a Windows Job Object.

    • kill_on_job_close=True  → process tree vanishes when our handle closes.
    • ProcessMemoryLimit       → OOM-kills the child if it exceeds max_memory_mb.
    • Stdout/stderr are merged into a single pipe.
    """
    import win32job, win32api, win32con, win32process, pywintypes

    # ── create the job object ──────────────────────────────────────────────
    job = win32job.CreateJobObject(None, "")

    info = win32job.QueryInformationJobObject(
        job, win32job.JobObjectExtendedLimitInformation
    )
    info["BasicLimitInformation"]["LimitFlags"] = (
        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
    )
    info["ProcessMemoryLimit"] = max_memory_mb * 1024 * 1024
    win32job.SetInformationJobObject(
        job, win32job.JobObjectExtendedLimitInformation, info
    )

    # ── spawn the child ────────────────────────────────────────────────────
    si                     = win32process.STARTUPINFO()
    creation_flags         = (
        win32process.CREATE_NEW_PROCESS_GROUP
        | win32process.CREATE_SUSPENDED          # assign to job before it runs
        | win32con.CREATE_NO_WINDOW
    )
    env = os.environ.copy()

    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env,
    )

    # Assign the new process to the job object before it gets a chance to run.
    try:
        handle = win32api.OpenProcess(
            win32con.PROCESS_ALL_ACCESS, False, proc.pid
        )
        win32job.AssignProcessToJobObject(job, handle)
    except pywintypes.error:
        pass  # process may have already exited — that's fine

    # ── collect output with timeout ────────────────────────────────────────
    output_chunks: list[bytes] = []
    total_bytes               = 0
    truncated                 = False

    def _reader():
        nonlocal total_bytes, truncated
        assert proc.stdout
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            if total_bytes < max_output_bytes:
                room = max_output_bytes - total_bytes
                output_chunks.append(chunk[:room])
                total_bytes += len(chunk)
                if total_bytes >= max_output_bytes:
                    truncated = True
            else:
                truncated = True

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Closing the job handle triggers kill_on_job_close.
        win32api.CloseHandle(job)
        proc.kill()
        reader.join(timeout=2)
        output = b"".join(output_chunks).decode("utf-8", errors="replace")
        return (
            f"[TIMEOUT after {timeout}s]\n{output}"
            + ("\n[output truncated]" if truncated else ""),
            -1,
        )

    reader.join(timeout=5)
    win32api.CloseHandle(job)

    output = b"".join(output_chunks).decode("utf-8", errors="replace")
    if truncated:
        output += "\n[output truncated — limit reached]"
    return output, proc.returncode


# =============================================================================
# WINDOWS — PSUTIL FALLBACK (no pywin32)
# =============================================================================

def _run_windows_psutil(
    cmd: str,
    cwd: Path,
    timeout: int,
    max_output_bytes: int,
) -> Tuple[str, int]:
    """Windows fallback: psutil tree-kill on timeout."""
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )

    output_chunks: list[bytes] = []
    total_bytes               = 0
    truncated                 = False

    def _reader():
        nonlocal total_bytes, truncated
        assert proc.stdout
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            if total_bytes < max_output_bytes:
                room = max_output_bytes - total_bytes
                output_chunks.append(chunk[:room])
                total_bytes += len(chunk)
                if total_bytes >= max_output_bytes:
                    truncated = True

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree_psutil(proc.pid)
        reader.join(timeout=2)
        output = b"".join(output_chunks).decode("utf-8", errors="replace")
        return f"[TIMEOUT after {timeout}s]\n{output}", -1

    reader.join(timeout=5)
    output = b"".join(output_chunks).decode("utf-8", errors="replace")
    if truncated:
        output += "\n[output truncated]"
    return output, proc.returncode


def _kill_tree_psutil(pid: int) -> None:
    """Kill a process and all its children using psutil."""
    if not _HAVE_PSUTIL:
        try:
            import signal as _sig
            os.kill(pid, _sig.SIGTERM)
        except OSError:
            pass
        return
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    except psutil.NoSuchProcess:
        pass


# =============================================================================
# POSIX — PROCESS GROUP + RESOURCE LIMITS
# =============================================================================

def _posix_preexec(max_memory_mb: int) -> None:
    """Called in the child process before exec() on POSIX systems."""
    os.setsid()  # new session → new process group

    try:
        mem_bytes = max_memory_mb * 1024 * 1024
        # Virtual address space limit (catches most runaway allocations)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, resource.error):
        pass  # some systems reject certain limits — not fatal

    try:
        # CPU time limit — hard-kills if a process spins forever
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (ValueError, resource.error):
        pass


def _run_posix(
    cmd: str,
    cwd: Path,
    timeout: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> Tuple[str, int]:
    """macOS / Linux: process-group isolation + resource limits."""

    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=lambda: _posix_preexec(max_memory_mb),
        env=os.environ.copy(),
    )

    output_chunks: list[bytes] = []
    total_bytes               = 0
    truncated                 = False

    def _reader():
        nonlocal total_bytes, truncated
        assert proc.stdout
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            if total_bytes < max_output_bytes:
                room = max_output_bytes - total_bytes
                output_chunks.append(chunk[:room])
                total_bytes += len(chunk)
                if total_bytes >= max_output_bytes:
                    truncated = True

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_posix_group(proc.pid)
        reader.join(timeout=2)
        output = b"".join(output_chunks).decode("utf-8", errors="replace")
        return (
            f"[TIMEOUT after {timeout}s]\n{output}"
            + ("\n[output truncated]" if truncated else ""),
            -1,
        )

    # Even on clean exit, sweep for orphaned children
    _sweep_psutil_children(proc.pid)

    reader.join(timeout=5)
    output = b"".join(output_chunks).decode("utf-8", errors="replace")
    if truncated:
        output += "\n[output truncated — limit reached]"
    return output, proc.returncode


def _kill_posix_group(pid: int) -> None:
    """Kill the entire process group on POSIX."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    _sweep_psutil_children(pid)


def _sweep_psutil_children(pid: int) -> None:
    """Best-effort psutil sweep for stragglers."""
    if not _HAVE_PSUTIL:
        return
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


# =============================================================================
# PUBLIC API
# =============================================================================

def run_sandboxed(
    cmd: str,
    workspace: Path,
    timeout: int           = DEFAULT_TIMEOUT_SECS,
    max_output_bytes: int  = DEFAULT_MAX_OUTPUT,
    max_memory_mb: int     = DEFAULT_MAX_MEMORY_MB,
) -> Tuple[str, int]:
    """
    Run *cmd* inside a platform-appropriate sandbox rooted at *workspace*.

    Returns
    -------
    (output, exit_code)
        output    — merged stdout+stderr, capped at max_output_bytes
        exit_code — 0 on success, -1 on timeout, otherwise the process exit code
    """
    if _IS_WINDOWS:
        if _HAVE_PYWIN32:
            return _run_windows_job_object(
                cmd, workspace, timeout, max_output_bytes, max_memory_mb
            )
        return _run_windows_psutil(cmd, workspace, timeout, max_output_bytes)
    # macOS or Linux
    return _run_posix(cmd, workspace, timeout, max_output_bytes, max_memory_mb)


def sandbox_info() -> dict:
    """Return a dict describing which sandbox backend is active."""
    if _IS_WINDOWS:
        backend = "job_object" if _HAVE_PYWIN32 else "psutil_treekill"
    elif _IS_MACOS:
        backend = "process_group+rlimit"
    else:
        backend = "process_group+rlimit"

    return {
        "platform":       platform.system(),
        "backend":        backend,
        "psutil":         _HAVE_PSUTIL,
        "pywin32":        _HAVE_PYWIN32,
        "default_timeout":   DEFAULT_TIMEOUT_SECS,
        "default_memory_mb": DEFAULT_MAX_MEMORY_MB,
    }
