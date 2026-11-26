#!/usr/bin/env python3
"""Strict prediction shape, identifiers and finite interval validation."""
import csv,math,sys
from pathlib import Path

def rows(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))
def validate(pred,input_dir):
    expected=rows(Path(input_dir)/"index.csv");data=rows(pred)
    ids=[r["id"] for r in expected]
    if len(data)!=len(ids) or len({r.get("id") for r in data})!=len(ids) or {r.get("id") for r in data}!=set(ids):raise ValueError("prediction IDs must match every input exactly once")
    for row in data:
        x=[float(row[k]) for k in ("estimate","lower","upper")]
        if not all(map(math.isfinite,x)):raise ValueError("nonfinite prediction")
        if not x[1]<=x[0]<=x[2]:raise ValueError("interval must contain estimate")
    return data
if __name__=="__main__":
    validate(sys.argv[1],sys.argv[2]);print("ok")
