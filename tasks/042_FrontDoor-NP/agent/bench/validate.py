#!/usr/bin/env python3
"""Validate one finite estimate and interval per observed dataset."""
import csv
import math
import sys


def read_predictions(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("no prediction rows")
    seen = set()
    for row in rows:
        if set(row) != {"case_id", "estimate", "lower", "upper"}:
            raise ValueError("prediction columns must be case_id,estimate,lower,upper")
        key = row["case_id"]
        if not key or key in seen:
            raise ValueError("empty or duplicate case_id")
        seen.add(key)
        for name in ("estimate", "lower", "upper"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]):
                raise ValueError("prediction values must be finite")
        if row["lower"] > row["upper"]:
            raise ValueError("reversed confidence interval")
    return rows


def validate(pred, inputs):
    rows = read_predictions(pred)
    with open(inputs, newline="") as f:
        expected = {r["case_id"] for r in csv.DictReader(f)}
    if {r["case_id"] for r in rows} != expected:
        raise ValueError("prediction IDs do not match input IDs")
    return rows


if __name__ == "__main__":
    try:
        rows = validate(*sys.argv[1:])
        print(f"ok: {len(rows)} datasets")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
