#!/usr/bin/env python3
"""MSE of the ATE estimates, guarded by nominal-CI coverage >= 0.887."""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import read_predictions


def score(pred, labels):
    rows = read_predictions(pred)
    with open(labels, newline="") as f:
        truth_rows = list(csv.DictReader(f))
    truth = {r["case_id"]: float(r["ate"]) for r in truth_rows}
    if len(truth) != len(truth_rows) or set(truth) != {r["case_id"] for r in rows}:
        raise ValueError("label/prediction IDs must be unique and identical")
    if any(not math.isfinite(v) for v in truth.values()):
        raise ValueError("nonfinite ground truth")
    errors = [(r["estimate"] - truth[r["case_id"]]) ** 2 for r in rows]
    # Strict inequalities reproduce the authors' organize_TMLE.R coverage formula.
    coverage = sum(r["lower"] < truth[r["case_id"]] < r["upper"] for r in rows) / len(rows)
    mse = math.fsum(errors) / len(rows)
    return {"mse": mse if coverage >= 0.887 else float("inf"),
            "unconstrained_mse": mse, "coverage": coverage,
            "coverage_pass": coverage >= 0.887, "n_datasets": len(rows)}


if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:])))
