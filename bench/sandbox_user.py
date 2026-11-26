#!/usr/bin/env python3
"""Run one sandbox command as a uid of its own (root helper, started as `sudo -n python3 sandbox_user.py ...`).

Usage: sandbox_user.py [--rw PATH]... [--own PATH]... [--rlimit NAME=SOFT:HARD]... [--account DIR --home PATH] -- COMMAND [ARG ...]

bwrap cannot give a sandbox its own PID namespace here (the container masks /proc, so a fresh /proc cannot be mounted),
so every process of the invoking user is visible and signalable from inside. Under a uid of its own, a sandbox can still
see other processes but cannot signal them, and nothing else can be signalled from inside it.
  1. take a free uid from UID_BASE.. (flock on /tmp/arb-uid/<uid>, held until exit; leftover processes of it are killed)
  2. chown the --rw trees and --own files to it (symlinks are not followed); with --account, write DIR/passwd and DIR/group:
     the host's entries plus one for this uid with home --home, for the sandbox to bind over /etc/passwd and /etc/group
     (getpass.getuser(), as in snakemake or torch's caches, raises KeyError for a uid without an entry)
  3. run COMMAND as that uid and gid with no supplementary groups and the caller's resource limits (sudo lowers open
     files to 1024; its hard core limit of 0 cannot be raised); SIGTERM/SIGINT/SIGHUP received are passed on to it
  4. when it ends, SIGKILL every process left with that uid, then chown everything back to SUDO_UID:SUDO_GID
Exit status is the command's; a command killed by a signal makes this helper die by the same signal.
"""
import fcntl
import os
import resource
import signal
import subprocess
import sys
import time

UID_BASE = 3_000_000
UID_COUNT = 1000
LOCK_DIR = "/tmp/arb-uid"
_pending = []
_child = None


def _forward(signum, _frame):
    if _child is not None and _child.returncode is None:
        try:
            os.kill(_child.pid, signum)
        except ProcessLookupError:
            pass
    else:
        _pending.append(signum)


def _uids(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.startswith("Uid:"):
                    return {int(x) for x in line.split()[1:]}
    except OSError:
        pass
    return set()


def kill_uid(uid, timeout=30.0):
    """SIGKILL every process that has uid as a real, effective, saved or fs uid, until none is left. -> count."""
    me = os.getpid()
    n = 0
    deadline = time.monotonic() + timeout
    while True:
        found = [int(d) for d in os.listdir("/proc") if d.isdigit() and int(d) != me and uid in _uids(d)]
        if not found:
            return n
        for pid in found:
            try:
                os.kill(pid, signal.SIGKILL)
                n += 1
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"processes of uid {uid} survive SIGKILL: {found[:10]}")
        time.sleep(0.05)


def take_uid():
    """-> (uid, lock fd). The lock is released when this process exits, however it exits."""
    os.makedirs(LOCK_DIR, mode=0o755, exist_ok=True)
    for uid in range(UID_BASE, UID_BASE + UID_COUNT):
        fd = os.open(f"{LOCK_DIR}/{uid}", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        kill_uid(uid)   # left by a helper that was killed before its cleanup
        return uid, fd
    raise RuntimeError(f"no free sandbox uid in {UID_BASE}..{UID_BASE + UID_COUNT - 1}")


def checked(path):
    real = os.path.realpath(path)
    if not os.path.isabs(path) or real != os.path.abspath(path) or len(real.strip("/").split("/")) < 3:
        raise ValueError(f"refusing to chown {path!r}: needs an absolute, symlink-free path at least 3 levels deep")
    if not os.path.lexists(path):
        raise FileNotFoundError(path)
    parent = os.path.dirname(real)
    while parent != "/":
        if not os.stat(parent).st_mode & 0o001:
            raise PermissionError(f"{parent} is not searchable by other users, so the sandbox uid cannot reach {path}")
        parent = os.path.dirname(parent)
    return path


def chown_tree(path, uid, gid):
    os.lchown(path, uid, gid)
    if os.path.isdir(path) and not os.path.islink(path):
        for root, dirs, files in os.walk(path):   # does not descend into symlinked directories
            for name in dirs + files:
                try:
                    os.lchown(os.path.join(root, name), uid, gid)
                except FileNotFoundError:
                    pass


def write_account(dirpath, uid, home, owner):
    for name, line in (("passwd", f"sandbox:x:{uid}:{uid}:bench sandbox:{home}:/bin/bash\n"), ("group", f"sandbox:x:{uid}:\n")):
        with open(f"/etc/{name}", encoding="utf-8", errors="surrogateescape") as f:
            text = f.read()
        tmp = os.path.join(dirpath, f".{name}.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(text + ("" if not text or text.endswith("\n") else "\n") + line)
        os.chmod(tmp, 0o644)
        os.chown(tmp, *owner)
        os.replace(tmp, os.path.join(dirpath, name))   # replaces a symlink there instead of following it


def parse(argv):
    rw, own, limits, account = [], [], [], {}
    i = 0
    while i < len(argv) and argv[i] != "--":
        if argv[i] not in ("--rw", "--own", "--rlimit", "--account", "--home") or i + 1 >= len(argv):
            raise SystemExit(__doc__)
        if argv[i] == "--rlimit":
            name, values = argv[i + 1].split("=")
            limits.append((getattr(resource, name), tuple(int(v) for v in values.split(":"))))
        elif argv[i] == "--account":
            account["dir"] = checked(argv[i + 1])
        elif argv[i] == "--home":
            account["home"] = argv[i + 1]
        else:
            (rw if argv[i] == "--rw" else own).append(checked(argv[i + 1]))
        i += 2
    if i >= len(argv) - 1 or set(account) not in (set(), {"dir", "home"}):
        raise SystemExit(__doc__)
    return rw, own, limits, account, argv[i + 1:]


def main():
    global _child
    if os.geteuid() != 0:
        raise SystemExit("sandbox_user.py must run as root (via sudo -n)")
    rw, own, limits, account, cmd = parse(sys.argv[1:])
    for which, (soft, hard) in limits:   # back to the caller's, as far as the hard limit sudo left allows (core stays 0)
        cap = resource.getrlimit(which)[1]
        fit = lambda v: v if cap == resource.RLIM_INFINITY or v != resource.RLIM_INFINITY and v <= cap else cap
        resource.setrlimit(which, (fit(soft), fit(hard)))
    caller_uid, caller_gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _forward)
    uid, _lock = take_uid()
    try:
        for p in rw + own:
            chown_tree(p, uid, uid)
        if account:
            write_account(account["dir"], uid, account["home"], (caller_uid, caller_gid))
        _child = subprocess.Popen(cmd, user=uid, group=uid, extra_groups=[])
        for signum in _pending:
            os.kill(_child.pid, signum)
        while True:
            try:
                rc = _child.wait()
                break
            except InterruptedError:
                continue
    finally:
        try:
            kill_uid(uid)
        finally:
            for p in rw + own:
                if os.path.lexists(p):
                    chown_tree(p, caller_uid, caller_gid)
    if rc < 0:
        if -rc != signal.SIGKILL:   # SIGKILL's handler cannot be set (EINVAL); it is always the default
            signal.signal(-rc, signal.SIG_DFL)
        os.kill(os.getpid(), -rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
