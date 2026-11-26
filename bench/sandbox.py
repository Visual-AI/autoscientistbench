"""bwrap workspace closure used for the agent run AND for the judge's train/predict stages.

What the probe on this devbox / the merlin image allows (2026-10-27): --unshare-net works, loopback comes up,
/etc/hosts can be replaced, unix sockets bound into the sandbox cross the netns; --unshare-pid together with a
fresh /proc does not, ports < 1024 cannot be bound (uid stays 1000). Hence:
Why no PID namespace (measured 2026-11-07): user namespaces do work (that is how unprivileged bwrap runs here), but
mounting a procfs inside one is refused -- the outer container masks paths under /proc, and the kernel then forbids a
new procfs mount that would reveal them; `bwrap --unshare-pid --proc /proc` fails with EPERM on the proc mount. Even
root cannot do it instead: the container's capability bounding set has no CAP_SYS_ADMIN (`sudo unshare --pid`, `sudo
unshare --mount`, `sudo bwrap --unshare-pid` all fail at once), so hidepid=2 cannot be mounted either. --unshare-pid
with the host /proc bind-mounted is the one combination that runs, and it makes outside processes unsignalable, but
procps inside dies with "fatal library error, lookup self" and /proc/<pid of an own child> does not exist, which
breaks ps, pgrep and psutil for the agent and for task code. Hence: uid per sandbox, and nothing more.
  * network: --unshare-net; the only way out is 127.0.0.1:<port> -> bridge.py -> unix socket -> proxy.py
  * cpu: taskset -c <cpuset> on the bwrap process + thread env vars (cgroup v2 is not writable here)
  * processes: without a PID namespace every process of the user is visible and signalable from inside, so a sandbox
        that runs agent code gets a uid of its own (own_uid=True): `sudo -n` + sandbox_user.py chown its writable
        binds to that uid, run it, kill what is left of that uid and chown back. ARB_SANDBOX_UID=0 turns this off.
        /etc/passwd and /etc/group inside get an entry for that uid (getpass.getuser() fails without one).
  * fs: explicit ro-binds of the system dirs, the workspace, the venv, the bench sources and the tool binaries;
        /mnt is never bound, so the NAS (full datasets, test numbers) does not exist for the agent.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

from . import LOCAL_ROOT, ROOT
from .util import fmt_cpuset

SYSTEM_DIRS = ["/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc"]
IN_SRC = "/opt/arb/src"      # autoresearch-bench/ (ro)
IN_BIN = "/opt/arb/bin"      # tool binaries + `bench` launcher (ro)
IN_RUN = "/run/arb"          # BENCH_RUN_DIR: state.json (ro file), submissions/, events.jsonl, proxy.sock, entry.sh
IN_WS = "/ws"
IN_VENV = "/venv"
IN_HOME = "/home/agent"
IN_TMP = "/tmp"
IN_UVCACHE = "/uvcache"


def bwrap_bin():
    return os.environ.get("ARB_BWRAP", shutil.which("bwrap") or "bwrap")


_SUDO_OK = None


def own_uid_prefix(rw, own, account_dir, home):
    """argv prefix that runs the rest as a uid of its own (see sandbox_user.py); [] when ARB_SANDBOX_UID=0.
    The helper writes account_dir/passwd and account_dir/group with an entry for that uid (home: its home inside)."""
    global _SUDO_OK
    if os.environ.get("ARB_SANDBOX_UID", "1") == "0":
        return []
    sudo = shutil.which("sudo") or "/usr/bin/sudo"
    if _SUDO_OK is None:
        try:
            _SUDO_OK = subprocess.run([sudo, "-n", "true"], capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _SUDO_OK = False
    if not _SUDO_OK:
        raise RuntimeError("sandboxes run as a uid of their own and need passwordless sudo (ARB_SANDBOX_UID=0 turns that off)")
    args = [sudo, "-n", sys.executable, str(Path(__file__).with_name("sandbox_user.py"))]
    for p in rw:
        args += ["--rw", str(p)]
    for p in own:
        args += ["--own", str(p)]
    args += ["--account", str(account_dir), "--home", home]
    for name in sorted(n for n in dir(resource) if n.startswith("RLIMIT_")):   # sudo resets some; the sandbox keeps ours
        soft, hard = resource.getrlimit(getattr(resource, name))
        args += ["--rlimit", f"{name}={soft}:{hard}"]
    return args + ["--"]


def _system_binds():
    args = []
    for d in SYSTEM_DIRS:
        if not os.path.exists(d):
            continue
        if os.path.islink(d):
            args += ["--symlink", os.readlink(d), d]
        else:
            args += ["--ro-bind", d, d]
    return args


def probe_proc_flags(cache_path=None):
    """Which /proc option works here. Cached in LOCAL_ROOT/.bwrap_probe.json."""
    cache_path = Path(cache_path or (LOCAL_ROOT / ".bwrap_probe.json"))
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())["proc"]
        except Exception:
            pass
    choice = None
    for cand in (["--proc", "/proc"], ["--ro-bind", "/proc", "/proc"], []):
        cmd = [bwrap_bin(), *_system_binds(), "--dev", "/dev", "--tmpfs", "/tmp", "--unshare-net", "--die-with-parent",
               *cand, "--", "/bin/sh", "-c", "exit 0"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        if r.returncode == 0:
            choice = cand
            break
    if choice is None:
        raise RuntimeError("bwrap cannot start a sandbox on this machine")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"proc": choice}))
    return choice


def tool_paths(names):
    """Resolve host paths of tool binaries (static musl builds of codex/traecli, uv) -> {name: realpath}.
    A name not on PATH is also looked for next to an already resolved tool (traecli ships traex-code-mode-host
    beside its own binary)."""
    out = {}
    for n in names:
        envk = "ARB_TOOL_" + n.upper().replace("-", "_")
        p = os.environ.get(envk) or shutil.which(n)
        if not p:
            for other in out.values():
                cand = os.path.join(os.path.dirname(other), n)
                if os.path.exists(cand):
                    p = cand
                    break
        if p:
            out[n] = os.path.realpath(p)
    return out


class Sandbox:
    """Builds the bwrap argv. Paths on the right of a bind are *inside* paths."""

    def __init__(self, cpuset=None, unshare_net=True, hostname_hosts=None, own_uid=False):
        self.binds = []       # ordered: (kind, host, inside); order matters (a later bind can cover an earlier one)
        self.own_uid = own_uid
        self.owned = []       # host files (sockets) the sandbox's uid must own while it runs
        self.env = {}
        self.cwd = IN_WS
        self.cpuset = list(cpuset) if cpuset else None
        self.unshare_net = unshare_net
        self.hosts_text = hostname_hosts or "127.0.0.1 localhost\n::1 localhost\n"
        self.proc_flags = None
        self.extra = []

    def ro_bind(self, host, inside):
        self.binds.append(("ro", str(host), inside))
        return self

    def rw_bind(self, host, inside):
        self.binds.append(("rw", str(host), inside))
        return self

    def ro_file(self, host, inside):
        self.binds.append(("ro", str(host), inside))
        return self

    def own_file(self, host):
        self.owned.append(str(host))
        return self

    def tmpfs(self, inside):
        self.binds.append(("tmpfs", None, inside))
        return self

    def setenv(self, **kw):
        self.env.update({k: str(v) for k, v in kw.items()})
        return self

    def add_tools(self, tools, launcher=None):
        """tools: {name: host_path}. launcher: host path of the `bench` script. All land in /opt/arb/bin."""
        for n, p in tools.items():
            self.ro_file(p, f"{IN_BIN}/{n}")
        if launcher:
            self.ro_file(launcher, f"{IN_BIN}/bench")
        # Agent-side imports only need bench/. Binding ROOT would also disclose
        # tasks/*/judge test labels and operator verification artifacts.
        self.ro_bind(ROOT / "bench", IN_SRC + "/bench")
        return self

    def argv(self, inner_cmd, scratch_dir):
        """scratch_dir: host dir where we may write the hosts/resolv files for this sandbox."""
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        hosts = scratch_dir / "hosts"
        hosts.write_text(self.hosts_text)
        resolv = scratch_dir / "resolv.conf"
        resolv.write_text("# no DNS inside the sandbox; only the allowlist proxy on 127.0.0.1\n")
        if self.proc_flags is None:
            self.proc_flags = probe_proc_flags()
        args = own_uid_prefix([h for kind, h, _ in self.binds if kind == "rw"], self.owned, scratch_dir,
                              self.env.get("HOME", "/")) if self.own_uid else []
        account = bool(args)
        if self.cpuset:
            args += ["taskset", "-c", fmt_cpuset(self.cpuset)]
        args += [bwrap_bin(), *_system_binds(), "--dev", "/dev", *self.proc_flags, "--die-with-parent"]
        if self.unshare_net:
            args += ["--unshare-net"]
        args += ["--ro-bind", str(hosts), "/etc/hosts", "--ro-bind", str(resolv), "/etc/resolv.conf"]
        if account:   # written by sandbox_user.py just before bwrap starts, with an entry for the sandbox's uid
            args += ["--ro-bind", str(scratch_dir / "passwd"), "/etc/passwd", "--ro-bind", str(scratch_dir / "group"), "/etc/group"]
        for kind, h, i in self.binds:
            if kind == "ro":
                args += ["--ro-bind", h, i]
            elif kind == "rw":
                args += ["--bind", h, i]
            else:
                args += ["--tmpfs", i]
        args += ["--clearenv"]
        for k, v in self.env.items():
            args += ["--setenv", k, v]
        args += ["--chdir", self.cwd, *self.extra, "--", *[str(c) for c in inner_cmd]]
        return args


def python_binds_for_venv(venv_dir):
    """Host dirs that must be visible inside so the venv's interpreter symlinks resolve.

    uv venvs point at a managed install twice: bin/python -> .../uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
    (a minor-version symlink dir) whose real path is .../cpython-3.11.15-linux-x86_64-gnu. Both paths are bound
    read-only at their own location; interpreters under the system dirs need nothing."""
    venv_dir = Path(venv_dir)
    cands = []
    py = venv_dir / "bin" / "python"
    try:
        cands.append(Path(os.path.realpath(py)).parent.parent)
    except OSError:
        pass
    cfg = venv_dir / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.startswith("home"):
                home = Path(line.split("=", 1)[1].strip())
                cands.append(home.parent)
    out = []
    for prefix in cands:
        sp = str(prefix)
        if any(sp == d or sp.startswith(d + "/") for d in SYSTEM_DIRS):
            continue
        if sp not in out and Path(sp).exists():
            out.append(sp)
    return out


BENCH_LAUNCHER = """#!/bin/sh
# `bench` inside the sandbox -> agent-side CLI of the bench package (stdlib only).
exec /usr/bin/python3 -s %(src)s/bench/agentcli.py "$@"
"""

ENTRY_SH = """#!/bin/sh
# sandbox entry: start the loopback->unix bridge, wait for it, then exec the harness command.
# The bridge runs under the name arb-netbridge so that the word "python" is not in its command line: an agent's
# own housekeeping kills it otherwise. DeepSeek-V4.1 ran `pkill -f "python3 -"` on 2026-11-10 to clean up its
# background scripts, which also matched `/usr/bin/python3 -s .../bridge.py`; with the bridge gone every model
# request failed ("stream disconnected ... 127.0.0.1:18080"), five reconnects, then crash -- 112 commands of work
# lost with nothing submitted. Seed's `pkill -9 -f python` killed its own harness the same hour.
# Only bash has `exec -a` (this script runs under dash, which errors with "exec: -a: not found"), so the rename
# goes through one transient `bash -c` that execs itself away; without bash we fall back to the plain launch.
# Measured after the rename: cmdline is `arb-netbridge -s .../bridge.py --listen ...` (no "python" anywhere),
# relaying works, and bridge.py's sys.argv is unchanged. The one side effect is `sys.executable` coming back
# empty, because CPython derives it from argv[0] and "arb-netbridge" is neither a path nor on PATH; the stdlib
# still loads from the compiled-in prefix. bridge.py imports only argparse/os/socket/sys/threading and never
# spawns anything, so nothing here reads sys.executable -- keep it that way, or pass the real path explicitly.
if command -v bash > /dev/null 2>&1; then
  bash -c 'exec -a arb-netbridge /usr/bin/python3 -s "$1" --listen "$2" --unix "$3" --ready "$4"' \\
    arb-netbridge %(src)s/bench/bridge.py 127.0.0.1:%(port)d %(sock)s %(ready)s &
else
  /usr/bin/python3 -s %(src)s/bench/bridge.py --listen 127.0.0.1:%(port)d --unix %(sock)s --ready %(ready)s &
fi
BRIDGE=$!
i=0
while [ ! -f %(ready)s ] && [ $i -lt 100 ]; do sleep 0.05; i=$((i+1)); done
exec "$@"
"""


PROFILE_SH = """# bench sandbox: /etc/profile replacement. Debian's profile resets PATH for login shells (codex runs `bash -lc`),
# which would hide the venv and /opt/arb/bin. Keep the PATH bench-run set up and nothing else.
if [ -n "$BENCH_PATH" ]; then export PATH="$BENCH_PATH"; fi
umask 022
"""


def write_profile(path):
    path = Path(path)
    path.write_text(PROFILE_SH)
    return path


def write_launcher(path):
    path = Path(path)
    path.write_text(BENCH_LAUNCHER % {"src": IN_SRC})
    path.chmod(0o755)
    return path


def write_entry(path, port, sock_inside, ready_inside):
    path = Path(path)
    path.write_text(ENTRY_SH % {"src": IN_SRC, "port": port, "sock": sock_inside, "ready": ready_inside})
    path.chmod(0o755)
    return path
