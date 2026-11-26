#!/usr/bin/env python3
"""Metric exactly as the paper / the authors' benchmark compute it: test RMSE on the raw target scale (USD),
RMSE = sqrt(mean((y - pred)^2)) over the label ids (== sqrt(sklearn.metrics.mean_squared_error(y, pred))).

Usage: scorer.py <pred_file> <labels_file>  -> prints {"rmse": value}

The structure guard of the score rule (model = <= 4 rank-1 products over the 8 raw features,
reconstruction error < 1e-6 relative; otherwise the score is +inf) cannot be decided from a prediction file.
It is implemented in `bench/check_model.py`, which the judge runs on the trained model directory.
"""
import csv
import json
import sys

import numpy as np


def read_csv(path, col):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = float(row[col])
    return out


def main(pred_path, labels_path):
    pred = read_csv(pred_path, "pred")
    lab = read_csv(labels_path, "y")
    ids = sorted(lab)
    missing = [i for i in ids if i not in pred]
    if missing:
        sys.exit(f"prediction file lacks {len(missing)} label ids (e.g. {missing[:3]})")
    y = np.array([lab[i] for i in ids], dtype=np.float64)
    p = np.array([pred[i] for i in ids], dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    print(json.dumps({"rmse": rmse}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
