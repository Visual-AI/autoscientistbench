#!/usr/bin/env python3
"""Run one command and reap its entire process tree, including detached workers.

Usage: python /absolute/path/process_supervisor.py -- COMMAND [ARG ...]
This short-lived Linux process owns the subreaper setting. It inherits the
caller's cwd, environment and standard streams without modifying the command.
"""
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time


_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_CLEANUP_SECONDS = 2.0
_termination_signal = 0


def _request_stop(signum, _frame):
    # A signal can arrive during cleanup. Never recurse into tree traversal or
    # interrupt a partially completed reap; the main loop observes this flag.
    global _termination_signal
    if not _termination_signal:
        _termination_signal = signum


def _enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    enabled = ctypes.c_int()
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise RuntimeError("child subreaper could not be verified")
    if _read_process(os.getpid()) is None:
        raise RuntimeError("/proc process ancestry is unavailable")


def _read_process(pid):
    """Return (parent PID, start ticks); start ticks protect against PID reuse."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as file:
            text = file.read()
        # The comm field can itself contain spaces and parentheses.
        fields = text[text.rfind(")") + 2:].split()
        return int(fields[1]), int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def _descendants():
    children = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                state = _read_process(pid)
            except (PermissionError, ValueError, IndexError):
                # Unrelated host processes need not be inspectable. Our
                # descendants retain the invoking uid and readable stat data.
                continue
            if state is not None:
                parent, birth = state
                children.setdefault(parent, []).append((pid, birth))
    found = {}
    pending = [os.getpid()]
    while pending:
        for pid, birth in children.get(pending.pop(), ()):
            if pid not in found:
                found[pid] = birth
                pending.append(pid)
    return found


def _signal_process(pid, birth, signum):
    """Signal precisely this process, using a pidfd where supported."""
    fd = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                fd = os.pidfd_open(pid)
            except OSError as exc:
                if exc.errno not in (errno.ENOSYS, errno.EINVAL):
                    raise
        state = _read_process(pid)
        if state is None or state[1] != birth:
            return
        if fd is not None:
            signal.pidfd_send_signal(fd, signum)
        else:
            os.kill(pid, signum)
    except ProcessLookupError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def _reap(leader):
    """Reap every available child; preserve the actual command's exit status."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False
        if pid == 0:
            return True
        if pid == leader.pid:
            leader.returncode = os.waitstatus_to_exitcode(status)


def _cleanup(leader):
    # Stop before killing, preventing the captured tree from continuing to
    # fork. Repeat discovery because a fork/reparent can race the first scan.
    deadline = time.monotonic() + _CLEANUP_SECONDS
    while True:
        descendants = _descendants()
        for pid, birth in descendants.items():
            _signal_process(pid, birth, signal.SIGSTOP)
        for pid, birth in descendants.items():
            _signal_process(pid, birth, signal.SIGKILL)
        children_remain = _reap(leader)
        if not descendants and not children_remain:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("descendants did not terminate after SIGKILL")
        time.sleep(.01)


def main(argv=None):
    global _termination_signal
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "--" or len(argv) < 2:
        print("usage: process_supervisor.py -- COMMAND [ARG ...]", file=sys.stderr)
        return 125
    _termination_signal = 0
    try:
        _enable_subreaper()
    except Exception as exc:
        print(f"process supervisor unavailable: {exc}", file=sys.stderr)
        return 125
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _request_stop)
    if _termination_signal:
        return -_termination_signal
    try:
        leader = subprocess.Popen(argv[1:])
    except OSError as exc:
        print(f"process supervisor could not start command: {exc}", file=sys.stderr)
        return 127 if exc.errno == errno.ENOENT else 126
    try:
        while leader.returncode is None and not _termination_signal:
            _reap(leader)
            if leader.returncode is None and not _termination_signal:
                time.sleep(.01)
    finally:
        try:
            _cleanup(leader)
        except Exception as exc:
            print(f"process supervisor cleanup failed: {exc}", file=sys.stderr)
            if leader.returncode is None:
                leader.kill()
            return 125
    return -_termination_signal if _termination_signal else leader.returncode


if __name__ == "__main__":
    result = main()
    if result < 0:
        # Preserve signal termination for Popen callers after cleanup completes.
        signum = -result
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    sys.exit(result)
