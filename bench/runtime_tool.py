#!/usr/bin/env python3
"""Run a trusted validator/scorer using the same copied prefixes as task code.

Keeping tools inside a separate sandbox also prevents absolute R installation
paths from silently falling back to an original reproduction on the host.
"""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import LOCAL_ROOT
from bench.budget import remaining_time
from bench.evalrun import stage_env
from bench.runtime import bind_runtime, load_snapshot, runtime_env
from bench.sandbox import Sandbox
from bench.task import load_task
from bench.util import free_cpus


def main():
    task = load_task(sys.argv[1])
    cpus = int(sys.argv[2])
    arguments = sys.argv[3:]
    deadline = os.environ.get("BENCH_EVAL_DEADLINE_TS")
    deadline = float(deadline) if deadline is not None else None
    remaining_time(deadline)
    base = LOCAL_ROOT / "runtime-tools" / task.pid
    base.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="call-", dir=base))
    (scratch / "tmp").mkdir()
    sb = Sandbox(cpuset=free_cpus()[:cpus], unshare_net=True)
    sb.tmpfs("/home/scorer").rw_bind(scratch / "tmp", "/tmp")
    bind_runtime(sb, task, deadline_ts=deadline)
    # Tools are trusted, but only their explicit input directories are visible.
    parents = set()
    for argument in arguments:
        try:
            path = Path(argument)
            if path.exists():
                path = path.resolve()
                parents.add(path if path.is_dir() else path.parent)
        except OSError:
            continue
    for parent in sorted(parents, key=lambda p: len(str(p))):
        sb.ro_bind(parent, str(parent))
    env = stage_env("/unused-venv", cpus, 0, str(task.data_dir), "/home/scorer", "/tmp",
                    extra={**runtime_env(task, deadline_ts=deadline), "BENCH_BENCH_DIR": str(task.judge_bench_dir)})
    if deadline is not None:
        remaining_time(deadline)
        env["BENCH_EVAL_DEADLINE_TS"] = str(deadline)
    # Trusted tools may import their frozen sibling modules.
    env.pop("PYTHONSAFEPATH", None)
    env.pop("VIRTUAL_ENV", None)
    sb.setenv(**env)
    sb.cwd = str(Path(arguments[0]).resolve().parent) if arguments and Path(arguments[0]).is_file() else "/tmp"
    argv = sb.argv([load_snapshot(task)["python"], *arguments], scratch)
    os.execvpe(argv[0], argv, os.environ)


if __name__ == "__main__":
    main()
