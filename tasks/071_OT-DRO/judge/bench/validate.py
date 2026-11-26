#!/usr/bin/env python3
"""Output-format check (same file the judge runs). Usage: validate.py <pred.csv> <input.npz>; exit 0 = ok.

Prediction = CSV with header exactly `id,w_star,L11,L21,L22,eps`, one row per input id (same id set, any order),
all values finite; eps > 0; L11 > 0 and L22 > 0 (L = [[L11, 0], [L21, L22]] must be a valid lower-triangular factor).
"""
import csv
import math
import sys

import numpy as np

COLS = ["id", "w_star", "L11", "L21", "L22", "eps"]


def main(pred_path, input_path):
    with np.load(input_path, allow_pickle=False) as z:
        ids = [str(i) for i in z["id"]]
    seen = {}
    with open(pred_path, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != COLS:
            sys.exit(f"header must be exactly {','.join(COLS)}; got {r.fieldnames}")
        for row in r:
            if None in row or any(v is None for v in row.values()):
                sys.exit("each row must have exactly six columns")
            iid = row["id"]
            if iid in seen:
                sys.exit(f"duplicate id {iid}")
            vals = {}
            for c in COLS[1:]:
                try:
                    v = float(row[c])
                except (TypeError, ValueError):
                    sys.exit(f"non-numeric {c} for id {iid!r}: {row.get(c)!r}")
                if math.isnan(v) or math.isinf(v):
                    sys.exit(f"NaN/inf {c} for id {iid}")
                vals[c] = v
            if vals["eps"] <= 0:
                sys.exit(f"eps must be > 0 for id {iid}: {vals['eps']}")
            if vals["L11"] <= 0 or vals["L22"] <= 0:
                sys.exit(f"L11 and L22 must be > 0 for id {iid}: {vals['L11']}, {vals['L22']}")
            seen[iid] = vals
    missing = [i for i in ids if i not in seen]
    extra = [i for i in seen if i not in set(ids)]
    if missing or extra:
        sys.exit(f"id mismatch: {len(missing)} missing, {len(extra)} extra (e.g. {missing[:3]} / {extra[:3]})")
    print(f"ok: {len(seen)} rows")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: validate.py <pred.csv> <input.npz>")
    main(sys.argv[1], sys.argv[2])
