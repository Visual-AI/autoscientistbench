"""Operator-owned, copied runtime snapshots for non-pip environments.

Snapshots live outside agent/code and are mounted read-only. Original installation
prefixes are aliases of the COPY, never binds of the reproduction directory.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from . import LOCAL_ROOT
from .budget import remaining_time
from .util import capture

IN_RUNTIME = "/opt/arb/runtime"
_FORBIDDEN_ENV = {"HOME", "TMPDIR", "TMP", "TEMP", "PYTHONHOME", "PYTHONSAFEPATH",
                  "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"}


def is_snapshot(task):
    return task.d["env"].get("mode") == "snapshot"


def load_snapshot(task):
    root = (task.dir / "runtime").resolve()
    path = (task.dir / task.d["env"].get("snapshot", "runtime/runtime.json")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("snapshot manifest must be inside task/runtime")
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("mounts"), list):
        raise ValueError("snapshot requires version=1 and a mounts list")
    mounts = []
    targets = set()
    for item in data["mounts"]:
        source = (task.dir / item["source"]).resolve()
        if not source.is_relative_to(root) or not source.exists():
            raise ValueError(f"snapshot source missing or outside runtime: {item['source']}")
        target = PurePosixPath(item["target"])
        # Only installation prefixes, never a whole host filesystem/workspace.
        if not target.is_absolute() or ".." in target.parts or len(target.parts) < 4:
            raise ValueError(f"unsafe snapshot target: {target}")
        if str(target).startswith(("/ws/", "/venv/", "/run/", "/home/agent/", "/home/judge/")):
            raise ValueError(f"snapshot target overlaps sandbox workspace: {target}")
        if str(target) in targets:
            raise ValueError(f"duplicate snapshot target: {target}")
        targets.add(str(target))
        fmt = item.get("format", "directory")
        if fmt not in ("directory", "tar"):
            raise ValueError(f"unsupported snapshot format: {fmt}")
        if (fmt == "directory" and not source.is_dir()) or (fmt == "tar" and not source.is_file()):
            raise ValueError(f"snapshot source has wrong type: {source}")
        name = item.get("name", source.name.removesuffix(".tar.gz").removesuffix(".tar"))
        if not name or "/" in name or name in (".", ".."):
            raise ValueError("snapshot mount name must be a basename")
        mounts.append(dict(item, source_path=source, target=str(target), name=name, format=fmt))
    for key, value in data.get("env", {}).items():
        if key in _FORBIDDEN_ENV or key.startswith("BENCH_") or not isinstance(value, str):
            raise ValueError(f"invalid snapshot environment entry: {key}")
    python = data.get("python", "/usr/bin/python3")
    if not isinstance(python, str) or not python.startswith("/"):
        raise ValueError("snapshot python must be an absolute executable path")
    if python not in ("/usr/bin/python3", "/usr/bin/python", "/usr/local/bin/python3") and not any(
            python.startswith(m["target"] + "/") or python.startswith(IN_RUNTIME + "/" + m["name"] + "/")
            for m in mounts):
        raise ValueError("snapshot python must be system Python or inside a copied mount")
    return dict(data, mounts=mounts, python=python)


def _extract(source, destination, deadline_ts=None):
    """Extract files first and symlinks last, preventing writes via archive links."""
    with tarfile.open(source, "r:*") as archive:
        links = []
        for member in archive:
            remaining_time(deadline_ts)
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe snapshot archive path: {member.name}")
            path = destination.joinpath(*rel.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, open(path, "wb") as dst:
                    while True:
                        remaining_time(deadline_ts)
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                path.chmod(member.mode & 0o777)
            elif member.issym() or member.islnk():
                links.append(member)
            else:
                raise ValueError(f"unsupported special file in snapshot: {member.name}")
        for member in links:
            remaining_time(deadline_ts)
            path = destination.joinpath(*PurePosixPath(member.name).parts)
            if not path.parent.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"archive link traverses another link: {member.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                path.symlink_to(member.linkname)
            else:
                rel = PurePosixPath(member.linkname)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"unsafe archive hardlink: {member.linkname}")
                target = destination.joinpath(*rel.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("archive hardlink escapes snapshot")
                os.link(target, path)


def materialize(task, deadline_ts=None):
    remaining_time(deadline_ts)
    data = load_snapshot(task)
    result = []
    for mount in data["mounts"]:
        remaining_time(deadline_ts)
        source = mount["source_path"]
        if mount["format"] == "directory":
            host = source
        else:
            stat = source.stat()
            key = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{mount.get('sha256', '')}".encode()).hexdigest()[:24]
            parent = LOCAL_ROOT / "snapshots" / task.pid
            host = parent / key
            if not (host / ".arb_snapshot_complete").is_file():
                parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=key + "-", dir=parent))
                try:
                    if mount.get("sha256"):
                        digest = hashlib.sha256()
                        with source.open("rb") as f:
                            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                                remaining_time(deadline_ts)
                                digest.update(chunk)
                        if digest.hexdigest() != mount["sha256"].removeprefix("sha256:"):
                            raise ValueError(f"snapshot digest mismatch: {source}")
                    _extract(source, staging, deadline_ts=deadline_ts)
                    _readable_by_all(staging)   # sandboxes run as uids of their own (sandbox_user.py); mkdtemp made it 0700
                    remaining_time(deadline_ts)
                    (staging / ".arb_snapshot_complete").write_text(key + "\n")
                    try:
                        staging.rename(host)
                    except OSError:
                        if not (host / ".arb_snapshot_complete").is_file():
                            raise
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
        result.append(dict(mount, host=host))
    remaining_time(deadline_ts)
    return dict(data, mounts=result)


def _readable_by_all(root):
    """chmod a+rX: directories readable and searchable by everyone, files readable, executables executable."""
    import stat
    pending = [Path(root)]
    while pending:
        path = pending.pop()
        # Some Python versions' rglob call is_dir() before is_symlink(),
        # resolving absolute runtime aliases into an unavailable NAS mount.
        # Snapshot permissions must never dereference a copied link.
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            continue
        directory = stat.S_ISDIR(mode)
        add = 0o555 if directory or mode & 0o111 else 0o444
        if mode & add != add:
            path.chmod(mode | add)
        if directory:
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)


def _host_value(value, data):
    pairs = []
    for mount in data["mounts"]:
        pairs += [(mount["target"], str(mount["host"])),
                  (IN_RUNTIME + "/" + mount["name"], str(mount["host"]))]
    # PATH/list entries and flags are supported, but never recursively rewrite replacements.
    import re
    mapping = dict(pairs)
    if mapping:
        pattern = "|".join(re.escape(s) for s in sorted(mapping, key=len, reverse=True))
        value = re.sub(pattern, lambda m: mapping[m.group()], value)
    return value


def runtime_env(task, host=False, deadline_ts=None):
    if not is_snapshot(task):
        return {}
    data = materialize(task, deadline_ts=deadline_ts)
    view = None
    if host:
        key = hashlib.sha256(json.dumps([(m["name"], str(m["host"])) for m in data["mounts"]]).encode()).hexdigest()[:16]
        view = LOCAL_ROOT / "snapshots" / task.pid / ("view-" + key)
        view.mkdir(parents=True, exist_ok=True)
        for mount in data["mounts"]:
            link = view / mount["name"]
            if not link.is_symlink():
                try:
                    link.symlink_to(mount["host"], target_is_directory=True)
                except FileExistsError:
                    if not link.is_symlink() or link.readlink() != mount["host"]:
                        raise
    env = dict(data.get("env", {}))
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    for key, value in env.items():
        value = value.replace("${BENCH_RUNTIME_DIR}", IN_RUNTIME)
        env[key] = _host_value(value, data).replace(IN_RUNTIME, str(view)) if host else value
    env["BENCH_RUNTIME_DIR"] = str(view) if host else IN_RUNTIME
    return env


def bind_runtime(sb, task, deadline_ts=None):
    if not is_snapshot(task):
        return
    for mount in materialize(task, deadline_ts=deadline_ts)["mounts"]:
        sb.ro_bind(mount["host"], mount["target"])
        stable = IN_RUNTIME + "/" + mount["name"]
        if stable != mount["target"]:
            sb.ro_bind(mount["host"], stable)


def host_python(task):
    data = materialize(task)
    return _host_value(data["python"], data)


def tool_launcher(task, directory, cpus):
    """Python-compatible operator launcher; explicit scorer inputs only, copied runtime."""
    path = Path(directory) / "snapshot-tool-python"
    path.parent.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name("runtime_tool.py")
    command = ["/usr/bin/python3", str(helper), str(task.dir.resolve()), str(cpus)]
    path.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n')
    path.chmod(0o755)
    return str(path)


def build_environment(task, venv_dir, lock_path, indexes, uv_cache, log_path, cap_s=1800):
    if not is_snapshot(task):
        from .envlock import build_venv
        return build_venv(venv_dir, task.python, lock_path, indexes, uv_cache, log_path, cap_s=cap_s)
    start = time.monotonic()
    deadline_ts = time.time() + float(cap_s)
    try:
        remaining_time(deadline_ts)
        # Snapshot environments are fixed: dependency edits must never be silently ignored.
        baseline = task.agent_dir / task.lock_rel
        if Path(lock_path).read_bytes() != baseline.read_bytes():
            raise ValueError("snapshot env.lock is immutable; runtime changes require a new operator snapshot")
        # Hashing and extracting a large snapshot must be interruptible even
        # while an individual filesystem/decompression call is blocked. A fresh
        # process also works when a suite builds environments from many threads.
        request = {"task_dir": str(task.dir.resolve()), "task": task.d,
                   "local_root": str(LOCAL_ROOT), "deadline_ts": deadline_ts}
        proc = capture([sys.executable, "-m", "bench.runtime"],
                       cwd=Path(__file__).resolve().parent.parent,
                       input_text=json.dumps(request), timeout=remaining_time(deadline_ts))
        if proc.returncode != 0:
            raise RuntimeError("snapshot materialization failed: " + proc.stderr[-2000:])
        data = json.loads(proc.stdout)
        if not data["ok"]:
            if data.get("timed_out"):
                raise TimeoutError(data["error"])
            raise ValueError(data["error"])
        remaining_time(deadline_ts)
        bin_dir = Path(venv_dir) / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        launcher = bin_dir / "python"
        launcher.write_text("#!/bin/sh\nexec " + shlex.quote(data["python"]) + ' "$@"\n')
        launcher.chmod(0o755)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"Copied runtime snapshot: {data['mount_count']} mounts; no install/reproduction performed.\n")
        remaining_time(deadline_ts)
        return {"ok": True, "elapsed_s": round(time.monotonic() - start, 2), "stage": "snapshot",
                "log": str(log_path), "timed_out": False}
    except Exception as exc:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(str(exc) + "\n")
        return {"ok": False, "elapsed_s": round(time.monotonic() - start, 2), "stage": "snapshot", "log": str(log_path),
                "timed_out": isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))}


def _materialize_worker():
    """Private operator subprocess; publish cache entries only after full extraction."""
    from .task import Task
    global LOCAL_ROOT
    request = json.load(sys.stdin)
    LOCAL_ROOT = Path(request["local_root"])
    try:
        data = materialize(Task(request["task_dir"], request["task"]), deadline_ts=request["deadline_ts"])
        result = {"ok": True, "python": data["python"], "mount_count": len(data["mounts"])}
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "timed_out": isinstance(exc, TimeoutError)}
    print(json.dumps(result))


if __name__ == "__main__":
    _materialize_worker()
