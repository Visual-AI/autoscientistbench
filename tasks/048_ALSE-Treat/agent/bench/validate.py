#!/usr/bin/env python3
"""Strict schema, IDs, learner pairs and finite ALSE standard errors."""
import csv
import json
import math
from pathlib import Path
import sys
COLUMNS=["case_id","n","learner","estimate","std_error"]
def expected(cases):
    ids=[c["case_id"] for c in cases]
    if not ids or len(ids)!=len(set(ids)) or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError("case IDs must be nonempty and unique")
    if any(type(c["n"]) is not int or c["n"]<1 for c in cases):
        raise ValueError("sample sizes must be positive integers")
    return {(c["case_id"],c["n"],learner) for c in cases for learner in ("gam","ranger")}
def read_predictions(path):
    rows={}
    with open(path,newline="") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=COLUMNS: raise ValueError("invalid prediction header")
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError("invalid row width")
            n=int(row["n"])
            key=(row["case_id"],n,row["learner"])
            if not key[0] or n<1 or key[2] not in ("gam","ranger") or key in rows:raise ValueError("invalid or duplicate key")
            estimate,se=float(row["estimate"]),float(row["std_error"])
            if not math.isfinite(estimate) or not math.isfinite(se) or se<0:raise ValueError("nonfinite estimate or invalid standard error")
            rows[key]=(estimate,se)
    if not rows:raise ValueError("empty predictions")
    return rows
def validate(pred,inputs):
    manifest=json.loads((Path(inputs)/"manifest.json").read_text())
    if manifest["format"]!="alse-observations-v1":raise ValueError("unknown input format")
    rows=read_predictions(pred)
    if set(rows)!=expected(manifest["cases"]):raise ValueError("predictions must match every input case and learner exactly")
    return rows
if __name__=="__main__":
    try:print("ok:",len(validate(*sys.argv[1:])),"predictions")
    except Exception as error:print(str(error),file=sys.stderr);sys.exit(1)
