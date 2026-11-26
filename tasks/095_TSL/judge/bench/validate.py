#!/usr/bin/env python3
"""Output-format check (same file the judge runs). Usage: validate.py <pred_file> <input_file>; exit 0 = ok."""
import csv
import math
import sys


def main(pred_path, input_path):
    with open(input_path, newline="") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    with open(pred_path, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "id" not in r.fieldnames or "pred" not in r.fieldnames:
            sys.exit(f"header must contain id,pred; got {r.fieldnames}")
        seen = {}
        for row in r:
            try:
                v = float(row["pred"])
            except (TypeError, ValueError):
                sys.exit(f"non-numeric pred for id {row.get('id')!r}: {row.get('pred')!r}")
            if math.isnan(v) or math.isinf(v):
                sys.exit(f"NaN/inf pred for id {row['id']}")
            if row["id"] in seen:
                sys.exit(f"duplicate id {row['id']}")
            seen[row["id"]] = v
    missing = [i for i in ids if i not in seen]
    extra = [i for i in seen if i not in set(ids)]
    if missing or extra:
        sys.exit(f"id mismatch: {len(missing)} missing, {len(extra)} extra (e.g. {missing[:3]} / {extra[:3]})")
    print(f"ok: {len(seen)} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
