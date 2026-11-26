"""task.json: load, defaults, validation, and the handful of derived paths everyone needs.

Layout of tasks/<pid>/ (DESIGN §3):
  task.json
  agent/   README.md  paper/  code/ (git repo, HEAD = adapt_commit, contains env.lock)  data/  bench/
  judge/   test/  reference/{env.lock,result.json,adapt.patch}  bench/ (authoritative copy of agent/bench)
"""
import copy
import json
import re
from pathlib import Path

from . import TASKS_DIR
from .catalog import retired_task_ids

PID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

DEFAULTS = {
    "class": "A",
    "title": "",
    "metric": {"name": None, "higher_is_better": None, "delta": 0.005, "aggregate": "mean"},
    "cpus": 4,
    "memory_gb": 16,
    "budget_hours_default": 5,
    "io": {
        "spec": "bench/io_spec.md",
        "validator": "bench/validate.py",
        "scorer": "bench/scorer.py",
        "val_input": "data/val/input",
        "val_labels": "data/val/labels",
        "test_input": "test/input",       # relative to judge/
        "test_labels": "test/labels",     # relative to judge/
    },
    "split": {"type": "std3", "decision": ""},
    "baseline": {"train_min": None, "eval_min": None, "val": {}, "paper_ref": ""},
    "judge": {
        "seeds": [1, 2, 3],
        "val_judge": True,
        "parity_rel_tol": 0.0,            # extra relative tolerance for reference parity (on top of sigma/2)
        "parity_abs_tol": 0.0,
    },
    "env": {"python": "3.11", "lock": "code/env.lock",
            "index": ["https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"],
            "build_cap_s": 1800},
    "adapt": {"base_commit": None, "adapt_commit": None, "patch": "judge/reference/adapt.patch"},
    "patch_limits": {"max_bytes": 2 * 1024 * 1024,
                     "forbidden_new_ext": [".csv", ".tsv", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
                                           ".h5", ".hdf5", ".parquet", ".feather", ".bin", ".zip", ".tar", ".gz",
                                           ".7z", ".xz", ".bz2", ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".so",
                                           ".safetensors", ".joblib", ".mat", ".arrow", ".db", ".sqlite"]},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Task:
    def __init__(self, task_dir, data):
        self.dir = Path(task_dir)
        self.raw = data
        self.d = _merge(DEFAULTS, data)
        self.pid = self.d.get("pid") or self.dir.name

    # ---- paths -------------------------------------------------------------
    @property
    def agent_dir(self):
        return self.dir / "agent"

    @property
    def judge_dir(self):
        return self.dir / "judge"

    @property
    def code_dir(self):
        return self.agent_dir / "code"

    @property
    def data_dir(self):
        return self.agent_dir / "data"

    @property
    def bench_dir(self):
        return self.agent_dir / "bench"

    @property
    def judge_bench_dir(self):
        return self.judge_dir / "bench"

    @property
    def reference_dir(self):
        return self.judge_dir / "reference"

    def val_input(self, agent_root=None):
        return Path(agent_root or self.agent_dir) / self.d["io"]["val_input"]

    def val_labels(self, agent_root=None):
        return Path(agent_root or self.agent_dir) / self.d["io"]["val_labels"]

    @property
    def test_input(self):
        return self.judge_dir / self.d["io"]["test_input"]

    @property
    def test_labels(self):
        return self.judge_dir / self.d["io"]["test_labels"]

    # ---- fields -------------------------------------------------------------
    @property
    def metric(self):
        return self.d["metric"]

    @property
    def metric_name(self):
        return self.d["metric"]["name"]

    @property
    def higher_is_better(self):
        return bool(self.d["metric"]["higher_is_better"])

    @property
    def cpus(self):
        return int(self.d["cpus"])

    @property
    def seeds(self):
        return [int(s) for s in self.d["judge"]["seeds"]]

    @property
    def train_cap_s(self):
        return None

    @property
    def predict_cap_s(self):
        return None

    @property
    def score_cap_s(self):
        return None

    @property
    def build_cap_s(self):
        return int(self.d["env"].get("build_cap_s") or 1800)

    @property
    def python(self):
        return str(self.d["env"]["python"])

    @property
    def indexes(self):
        return list(self.d["env"].get("index") or [])

    @property
    def lock_rel(self):
        return self.d["env"].get("lock") or "code/env.lock"

    @property
    def adapt_commit(self):
        return self.d["adapt"].get("adapt_commit")

    @property
    def base_commit(self):
        return self.d["adapt"].get("base_commit")

    @property
    def patch_limits(self):
        return self.d["patch_limits"]

    def index_hosts(self):
        from urllib.parse import urlparse
        hosts = []
        for u in self.indexes:
            h = urlparse(u).hostname
            if h and h not in hosts:
                hosts.append(h)
        # the simple index on pypi.org redirects wheels to files.pythonhosted.org
        if "pypi.org" in hosts and "files.pythonhosted.org" not in hosts:
            hosts.append("files.pythonhosted.org")
        return hosts

    # ---- validation ---------------------------------------------------------
    def problems(self, deep=False):
        p = []
        if self.d["env"].get("mode") == "snapshot":
            from .runtime import load_snapshot
            try:
                load_snapshot(self)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                p.append(f"runtime snapshot invalid (run `python3 -m bench fetch {self.pid}`): {exc}")
        if not PID_RE.match(self.pid):
            p.append(f"bad pid {self.pid!r}")
        m = self.d["metric"]
        if not m.get("name"):
            p.append("metric.name missing")
        if m.get("higher_is_better") is None:
            p.append("metric.higher_is_better missing")
        if not isinstance(self.d["judge"]["seeds"], list) or not self.d["judge"]["seeds"]:
            p.append("judge.seeds must be a non-empty list")
        if (self.code_dir / ".git").exists() and not self.d["adapt"].get("adapt_commit"):
            p.append("adapt.adapt_commit missing (run `bench adapt` / fixture)")
        for rel in ("README.md", "code", "bench", "data"):
            if not (self.agent_dir / rel).exists():
                hint = f" (run `python3 -m bench fetch {self.pid}`)" if rel == "data" else ""
                p.append(f"agent/{rel} missing{hint}")
        for rel in ("code/train.sh", "code/predict.sh", self.lock_rel, self.d["io"]["validator"], self.d["io"]["scorer"]):
            if not (self.agent_dir / rel).exists():
                p.append(f"agent/{rel} missing")
        if deep:
            if not self.test_input.exists():
                p.append(f"judge test input missing (run `python3 -m bench fetch {self.pid}`): {self.test_input}")
            if not self.test_labels.exists():
                p.append(f"judge test labels missing (run `python3 -m bench fetch {self.pid}`): {self.test_labels}")
            if not self.val_input().exists() or not self.val_labels().exists():
                p.append(f"agent val input/labels missing (run `python3 -m bench fetch {self.pid}`)")
        return p

    def to_json(self):
        return json.dumps(self.d, indent=2, ensure_ascii=False)


def load_task(pid_or_path):
    p = Path(pid_or_path)
    if p.is_dir() and (p / "task.json").exists():
        task_dir = p
    elif p.name == "task.json" and p.exists():
        task_dir = p.parent
    else:
        task_dir = TASKS_DIR / str(pid_or_path)
    tj = task_dir / "task.json"
    if not tj.exists():
        raise FileNotFoundError(f"no task.json at {tj}")
    with open(tj, encoding="utf-8") as f:
        data = json.load(f)
    return Task(task_dir, data)


def list_tasks():
    out = []
    retired = retired_task_ids()
    if TASKS_DIR.exists():
        for d in sorted(TASKS_DIR.iterdir()):
            if d.name not in retired and (d / "task.json").exists():
                out.append(d.name)
    return out
