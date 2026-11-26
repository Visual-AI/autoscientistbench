#!/usr/bin/env python3
"""Exact observed case matching and finite permutation p-values."""
import csv, math, sys
from pathlib import Path
COLUMNS = ["case_id", "p_value"]
def read_predictions(path):
    rows = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS: raise ValueError("invalid prediction columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()): raise ValueError("invalid row width")
            key = row["case_id"]
            if not key or key in rows: raise ValueError("empty or duplicate case ID")
            p = float(row["p_value"])
            if not math.isfinite(p) or not 0 <= p <= 1: raise ValueError("p-value must be finite and between 0 and 1")
            rows[key] = p
    if not rows: raise ValueError("empty predictions")
    return rows
def input_cases(inputs):
    with (Path(inputs)/"manifest.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["case_id", "file", "n"]: raise ValueError("invalid input manifest")
        rows = list(reader)
    ids = [row["case_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids): raise ValueError("invalid input IDs")
    if any(row["n"] != "100" or row["file"] != row["case_id"]+".rds" for row in rows): raise ValueError("invalid observed-data contract")
    return set(ids)
def validate(pred, inputs):
    rows = read_predictions(pred)
    if set(rows) != input_cases(inputs): raise ValueError("predictions must match every input case exactly")
    return rows
if __name__ == "__main__":
    try: print("ok:",len(validate(*sys.argv[1:])),"predictions")
    except Exception as exc: print(str(exc),file=sys.stderr); sys.exit(1)
