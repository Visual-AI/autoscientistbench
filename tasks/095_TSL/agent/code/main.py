#!/usr/bin/env python3
"""Bench entry points for the TSL (R<=2) baseline on california_housing: `train` and `predict`.

Protocol-only wrapper around the authors' package (`tensorsl`, built from tsl-py/ in this repository, tag v0.3.0):
  train   : read $BENCH_DATA_DIR/train/train.csv, fit `tensorsl.sklearn.TSLRegressor` with the authors' released
            best-trial hyperparameters (configs/california_housing_tsl_r2.json), seed = BENCH_SEED, and write
            DIR/model.bin (native serialisation), DIR/model_card.json (structure certificate, see model_card.py)
            and DIR/train_meta.json.
  predict : load DIR/model.bin, read the feature CSV given with --input (columns id + FEATURES), write `id,pred`.

The refit itself (model, hyperparameters, fitting loop) is exactly the authors' benchmark refit step; only the data
entry (files instead of an OpenML download + runtime split), the seed source and the save/load entry points are ours.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "california_housing_tsl_r2.json"
# OpenML dataset 44977 (CTR-23 task 361255) feature order; the target column is `y` (medianHouseValue, USD).
FEATURES = ["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households",
            "medianIncome"]


def read_table(path, with_y):
    """CSV with header id,<FEATURES>[,y] -> (ids, X float64 (n, 8), y or None)."""
    ids, X, y = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in FEATURES + (["y"] if with_y else []) if c not in (r.fieldnames or [])]
        if missing:
            sys.exit(f"{path}: missing columns {missing}; header is {r.fieldnames}")
        for row in r:
            ids.append(row["id"])
            X.append([float(row[c]) for c in FEATURES])
            if with_y:
                y.append(float(row["y"]))
    X = np.ascontiguousarray(np.array(X, dtype=np.float64))
    return ids, X, (np.array(y, dtype=np.float64) if with_y else None)


def train(a):
    import tensorsl
    from tensorsl.sklearn import TSLRegressor
    from model_card import export_card

    seed = int(os.environ.get("BENCH_SEED", a.seed))
    cfg = json.loads(CONFIG.read_text())
    params = cfg["best_params"]
    ids, X, y = read_table(Path(a.data) / "train" / "train.csv", True)
    print(f"[data]   {len(ids)} rows x {X.shape[1]} features from {Path(a.data) / 'train' / 'train.csv'}", flush=True)
    print(f"[config] {CONFIG.name} seed={seed} params={params}", flush=True)
    t0 = time.time()
    model = TSLRegressor(seed=seed, verbosity=1, **params).fit(X, y)
    fit_s = time.time() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "model.bin"))
    card = export_card(model, FEATURES)
    (out / "model_card.json").write_text(json.dumps(card) + "\n")
    pred_tr = np.asarray(model.predict(X), dtype=np.float64)
    meta = {"seed": seed, "n_train": len(ids), "fit_s": round(fit_s, 2), "n_stages": len(model.stage_predictors),
            "train_rmse": float(np.sqrt(np.mean((pred_tr - y) ** 2))), "params": params, "tensorsl": tensorsl.__version__}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[fit]    {fit_s:.1f}s stages={meta['n_stages']} train_RMSE={meta['train_rmse']:.4f}", flush=True)
    print(f"[saved]  {out / 'model.bin'}, {out / 'model_card.json'}, {out / 'train_meta.json'}", flush=True)


def predict(a):
    from tensorsl.sklearn import TSLRegressor

    model = TSLRegressor.load(str(Path(a.model) / "model.bin"))
    ids, X, _ = read_table(a.input, False)
    pred = np.asarray(model.predict(X), dtype=np.float64)
    with open(a.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred"])
        for i, p in zip(ids, pred):
            w.writerow([i, repr(float(p))])
    print(f"wrote {len(ids)} predictions to {a.output}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--data", required=True, help="$BENCH_DATA_DIR (contains train/train.csv)")
    t.add_argument("--seed", type=int, default=0, help="fallback when BENCH_SEED is not set")
    t.add_argument("--out", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)
