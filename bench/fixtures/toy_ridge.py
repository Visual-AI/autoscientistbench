"""toy-ridge: a synthetic regression task that trains in a second. Exists to test the harness end to end.
Baseline = ridge regression on raw features for a nonlinear target (deliberately underfit; poly/sin features win)."""
import csv
import math
import random
from pathlib import Path

from .. import TASKS_DIR
from ..convert import adapt, init
from ..util import read_json, write_json

PID = "toy-ridge"

MAIN_PY = r'''#!/usr/bin/env python3
"""Baseline: ridge regression (closed form). FEATURES / LAMBDA are the method knobs."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

FEATURES = "linear"
LAMBDA = 1.0
X_COLS = ["x1", "x2", "x3", "x4", "x5"]


def featurize(X):
    if FEATURES == "linear":
        return X
    if FEATURES == "poly_sin":
        n = X.shape[1]
        cols = [X, X ** 2, np.sin(2.5 * X), np.cos(2.5 * X)]
        cols += [X[:, i:i + 1] * X[:, j:j + 1] for i in range(n) for j in range(i + 1, n)]
        return np.hstack(cols)
    raise ValueError(FEATURES)


def read(path, with_y):
    ids, X, y = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            X.append([float(row[c]) for c in X_COLS])
            if with_y:
                y.append(float(row["y"]))
    return ids, np.array(X), (np.array(y) if with_y else None)


def train(a):
    rng = np.random.default_rng(int(os.environ.get("BENCH_SEED", a.seed)))
    _, X, y = read(Path(a.data) / "train" / "train.csv", True)
    F = featurize(X)
    mu, sd = F.mean(0), F.std(0) + 1e-8
    Z = (F - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    reg = LAMBDA * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "features": FEATURES, "lambda": LAMBDA,
               "seed_draw": float(rng.random())}, open(out / "model.json", "w"))
    print(f"trained ridge features={FEATURES} lambda={LAMBDA} n={len(y)} d={Z.shape[1]}")


def predict(a):
    m = json.load(open(Path(a.model) / "model.json"))
    global FEATURES
    FEATURES = m["features"]
    ids, X, _ = read(a.input, False)
    Z = (featurize(X) - np.array(m["mu"])) / np.array(m["sd"])
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    pred = Z @ np.array(m["w"])
    with open(a.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred"])
        for i, p in zip(ids, pred):
            w.writerow([i, f"{p:.6f}"])
    print(f"wrote {len(ids)} predictions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--seed", type=int, default=0); t.add_argument("--out", required=True)
    p = sp.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)
'''

README = """# toy-ridge: synthetic nonlinear regression (harness test task)

## Task
Predict `y` from five features `x1..x5`. The generator is y = sin(2.5 x1) + 0.8 x2^2 - 0.5 x3 + 0.3 x4 x5 + noise,
features uniform in [-2, 2], noise N(0, 0.1^2). Train 3000 rows, val 600 rows; the judge holds 1500 test rows from
the same generator with a different seed.

## Data
- `data/train/train.csv` — `id,x1,...,x5,y`
- `data/val/input.csv` (`id,x1..x5`) and `data/val/labels.csv` (`id,y`)

## Metric
`mse` (lower is better), `bench/scorer.py`. PASS threshold: 2 % relative improvement over the baseline.

## Baseline
`code/main.py`: ridge regression on the raw features (closed form, lambda = 1). It is linear, the target is not.

## Interface
- `bash code/train.sh --out DIR` -> writes `DIR/model.json`
- `bash code/predict.sh --model DIR --input PATH --output FILE` -> CSV `id,pred`
- env: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`

## Known pitfalls
Training must finish in 60 s (judge cap); keep it small. numpy is the only dependency in `code/env.lock`.
"""


def gen(n, seed):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        x = [rnd.uniform(-2, 2) for _ in range(5)]
        y = math.sin(2.5 * x[0]) + 0.8 * x[1] ** 2 - 0.5 * x[2] + 0.3 * x[3] * x[4] + rnd.gauss(0, 0.1)
        rows.append((f"r{i:05d}", x, y))
    return rows


def write_rows(path, rows, with_y, labels_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if labels_only:
            w.writerow(["id", "y"])
            for i, _, y in rows:
                w.writerow([i, f"{y:.6f}"])
            return
        w.writerow(["id", "x1", "x2", "x3", "x4", "x5"] + (["y"] if with_y else []))
        for i, x, y in rows:
            w.writerow([i] + [f"{v:.6f}" for v in x] + ([f"{y:.6f}"] if with_y else []))


def build(force=False, numpy_pin="numpy==2.2.6"):
    d = init(PID, title="Toy ridge regression (synthetic)", metric="mse", higher=False, python="3.11", cpus=2, force=force)
    write_rows(d / "agent/data/train/train.csv", gen(3000, 11), True)
    val = gen(600, 22)
    write_rows(d / "agent/data/val/input.csv", val, False)
    write_rows(d / "agent/data/val/labels.csv", val, False, labels_only=True)
    test = gen(1500, 33)
    write_rows(d / "judge/test/input.csv", test, False)
    write_rows(d / "judge/test/labels.csv", test, False, labels_only=True)
    (d / "agent/code/main.py").write_text(MAIN_PY)
    (d / "agent/code/env.lock").write_text(numpy_pin + "\n")
    (d / "agent/README.md").write_text(README)
    tj = read_json(d / "task.json")
    tj["metric"]["delta"] = 0.02
    tj["baseline"].update({"train_min": 0.5, "eval_min": 0.5})
    tj["env"]["index"] = ["https://pypi.org/simple"]
    tj["env"]["build_cap_s"] = 600
    tj["memory_gb"] = 4
    write_json(d / "task.json", tj)
    adapt(PID, message="baseline snapshot", allow_fresh=True)
    print(f"fixture {PID} built at {d}")
    return d
