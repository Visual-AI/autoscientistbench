#!/usr/bin/env python3
"""Validate finite estimates and consistent callback-ledger arithmetic.
The ledger is NOT authenticated against a malicious candidate; audit interface bypass.
"""
import csv
import math
import sys

FIELDS = {"case_id", "estimate", "reported_cost", "single", "coupled"}

def read_predictions(path):
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != FIELDS:
            raise ValueError("unexpected prediction columns")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("at least two independent estimates are required")
    seen = set()
    for row in rows:
        key = row["case_id"]
        if not key or key in seen:
            raise ValueError("empty or duplicate case ID")
        seen.add(key)
        for field in ("estimate", "reported_cost", "single", "coupled"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError("nonfinite prediction or counter")
        for field in ("single", "coupled"):
            if row[field] < 0 or not row[field].is_integer():
                raise ValueError("counter must be a nonnegative integer")
        row["counted_cost"] = int(row["single"]) + 2 * int(row["coupled"])
        if row["counted_cost"] <= 0 or row["reported_cost"] != row["counted_cost"]:
            raise ValueError("estimator cost and callback ledger disagree")
    return rows

def validate(pred, inputs):
    rows = read_predictions(pred)
    with open(inputs, newline="") as stream:
        observed = list(csv.DictReader(stream))
    expected = [r["case_id"] for r in observed]
    if len(expected) != len(set(expected)) or {r["case_id"] for r in rows} != set(expected):
        raise ValueError("prediction/input IDs differ")
    return rows

if __name__ == "__main__":
    try:
        print("ok:", len(validate(*sys.argv[1:])), "independent runs")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
