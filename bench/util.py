"""Small shared helpers: hashing, json/jsonl IO, subprocess with group kill and wall cap."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ts_compact():
    return time.strftime("%Y%m%d-%H%M%S")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            d = f.read(chunk)
            if not d:
                break
            h.update(d)
    return h.hexdigest()


def sha256_tree(root, rel_prefix=""):
    """Deterministic hash of a directory: sorted (relpath, sha256(file)) pairs."""
    root = Path(root)
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
            items.append(f"{rel_prefix}{p.relative_to(root).as_posix()}:{sha256_file(p)}")
    return sha256_bytes("\n".join(items).encode())


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def write_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return out


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def rm_rf(p):
    p = Path(p)
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
    elif p.exists():
        shutil.rmtree(p, ignore_errors=True)


def log(msg, file=sys.stderr):
    print(f"{time.strftime('%F %T')} {msg}", file=file, flush=True)


def kill_group(p, grace=15.0):
    """SIGTERM then SIGKILL the whole process group of a Popen started with start_new_session."""
    pg = p.pid
    try:
        os.killpg(pg, signal.SIGTERM)
    except OSError:
        return
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may exit while a child ignores SIGTERM. Still kill the
        # group so an expired evaluation cannot leave computation running.
        try:
            os.killpg(pg, signal.SIGKILL)
        except OSError:
            pass
    p.wait()


def kill_stragglers(root):
    """SIGKILL every process whose cwd is under root (left-over training runs). -> count."""
    root = os.path.realpath(str(root))
    me = os.getpid()
    n = 0
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cwd = os.path.realpath(os.readlink(f"/proc/{d}/cwd"))
        except OSError:
            continue
        if cwd == root or cwd.startswith(root + os.sep):
            try:
                os.kill(int(d), signal.SIGKILL)
                n += 1
            except OSError:
                pass
    return n


def supervised_argv(cmd):
    """A dedicated subreaper also owns descendants that create new sessions. Behind a sandbox_user.py prefix it goes
    after the prefix, so it runs as the sandbox's uid and can kill what it owns."""
    cmd = [str(c) for c in cmd]
    sup = [sys.executable, str(Path(__file__).with_name("process_supervisor.py")), "--"]
    if str(Path(__file__).with_name("sandbox_user.py")) in cmd[:4]:
        i = cmd.index("--") + 1
        return cmd[:i] + sup + cmd[i:]
    return sup + cmd


def run(cmd, cwd=None, env=None, timeout=None, log_path=None, stdin_text=None, append=True, echo=False):
    """Run cmd in its own process group; tee output to log_path.
    -> dict(rc, elapsed_s, timed_out, log)."""
    t0 = time.time()
    lf = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        lf = open(log_path, "a" if append else "w", encoding="utf-8")
        lf.write(f"\n### {now_iso()} cwd={cwd}\n### $ {' '.join(map(str, cmd))}\n")
        lf.flush()
    timed_out = False
    try:
        p = subprocess.Popen(supervised_argv(cmd), cwd=cwd, env=env, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                             stdout=lf or (None if echo else subprocess.DEVNULL), stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
    except OSError as e:
        if lf:
            lf.write(f"### spawn failed: {e}\n")
            lf.close()
        return {"rc": -98, "elapsed_s": 0.0, "timed_out": False, "log": str(log_path) if log_path else None, "error": str(e)}
    try:
        p.communicate(input=stdin_text, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        kill_group(p)
        rc, timed_out = -1, True
    finally:
        # Successful parent exit does not imply its worker children exited.
        kill_group(p)
    el = time.time() - t0
    if lf:
        lf.write(f"### rc={rc} elapsed={el:.1f}s{' TIMEOUT' if timed_out else ''}\n")
        lf.close()
    return {"rc": rc, "elapsed_s": round(el, 2), "timed_out": timed_out, "log": str(log_path) if log_path else None}


def capture(cmd, cwd=None, env=None, timeout=600, check=False, input_text=None):
    """Run and capture stdout/stderr (small outputs)."""
    argv = [str(c) for c in cmd]
    proc = subprocess.Popen(supervised_argv(argv), cwd=cwd, env=env, text=True, start_new_session=True,
                            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired as exc:
            # Bound pipe draining even if an unrelated external process holds
            # an inherited descriptor. Our own descendants belong to the subreaper.
            stdout, stderr = exc.output or "", exc.stderr or ""
            proc.stdout.close()
            proc.stderr.close()
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr)
    finally:
        kill_group(proc)
    p = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    return p


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0


def free_cpus():
    """CPU ids this process may use (respects taskset/cgroup cpuset, not the cfs quota)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def parse_cpuset(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def fmt_cpuset(cpus):
    return ",".join(str(c) for c in cpus)


def thread_env(n):
    """Thread caps for every numeric library we have met (from pipeline/runner.py)."""
    th = str(int(n))
    return {
        "OMP_NUM_THREADS": th, "MKL_NUM_THREADS": th, "OPENBLAS_NUM_THREADS": th, "NUMEXPR_MAX_THREADS": th,
        "NUMEXPR_NUM_THREADS": th, "VECLIB_MAXIMUM_THREADS": th, "RAYON_NUM_THREADS": th,
        "TF_NUM_INTRAOP_THREADS": th, "TF_NUM_INTEROP_THREADS": "2", "POLARS_MAX_THREADS": th,
        "LOKY_MAX_CPU_COUNT": th, "NUMBA_NUM_THREADS": th, "PYTHON_CPU_COUNT": th, "MAX_JOBS": th,
        "CMAKE_BUILD_PARALLEL_LEVEL": th, "UV_CONCURRENT_BUILDS": "2", "UV_CONCURRENT_INSTALLS": th,
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
