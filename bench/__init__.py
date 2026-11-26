"""AutoScientistBench — outer harness (see ../PROTOCOL.md).

One package, two faces:
  * operator side  : python -m bench {tasks,fetch,assets-status,lint,reference,run,judge,suite,...}
  * agent side     : `bench {status,eval-dev,validate-output,submit}` inside the sandbox (agentcli.py)
All task-specific knowledge lives in tasks/<pid>/ (task.json, bench/scorer.py, bench/validate.py,
code/train.sh, code/predict.sh); nothing in this package branches on a task id.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # autoscientistbench/
TASKS_DIR = Path(os.environ.get("ARB_TASKS_DIR", ROOT / "tasks"))
RUNS_DIR = Path(os.environ.get("ARB_RUNS_DIR", ROOT / "runs"))
RESULTS_DIR = Path(os.environ.get("ARB_RESULTS_DIR", ROOT / "results"))
HARNESSES_DIR = Path(os.environ.get("ARB_HARNESSES_DIR", ROOT / "harnesses"))
# node-local scratch (workspaces, venvs, judge clones); prefer fast local disk
LOCAL_ROOT = Path(os.environ.get("ARB_LOCAL_ROOT", "/tmp/arb"))
# trusted uv cache for operator/judge builds (override with ARB_UV_CACHE)
UV_CACHE = Path(os.environ.get("ARB_UV_CACHE", LOCAL_ROOT / "uv-cache"))
VERSION = "0.1.0"
