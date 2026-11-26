#!/usr/bin/env python3
"""Validate exact setting/replicate coverage and finite estimator intervals."""
import csv
import json
import math
import sys
from pathlib import Path

COLUMNS = ["setting_id", "replicate", "estimate", "lower", "upper"]

def read_predictions(path):
    rows = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("prediction header must be " + ",".join(COLUMNS))
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid row width")
            replicate = int(row["replicate"])
            key = (row["setting_id"], replicate)
            if not key[0] or replicate < 1 or key in rows:
                raise ValueError("empty/duplicate setting or invalid replicate")
            values = tuple(float(row[name]) for name in COLUMNS[2:])
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite prediction")
            estimate, lower, upper = values
            if not lower <= estimate <= upper:
                raise ValueError("confidence interval must enclose point estimate")
            rows[key] = values
    if not rows:
        raise ValueError("empty predictions")
    return rows

def expected_keys(settings):
    ids = [setting["setting_id"] for setting in settings]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("duplicate/empty setting IDs")
    return {(setting["setting_id"], replicate)
            for setting in settings for replicate in range(1, int(setting["replications"]) + 1)}

def validate(pred, inputs):
    with open(Path(inputs) / "manifest.json") as handle:
        manifest = json.load(handle)
    if manifest["format"] != "acbc-observations-v1":
        raise ValueError("unsupported input format")
    rows = read_predictions(pred)
    if set(rows) != expected_keys(manifest["settings"]):
        raise ValueError("predictions must cover every setting and replicate exactly once")
    return rows

if __name__ == "__main__":
    try:
        print("ok:", len(validate(*sys.argv[1:])), "predictions")
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
