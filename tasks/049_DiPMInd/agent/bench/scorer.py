#!/usr/bin/env python3
"""Exact manifest mean-power metric and both empirical-size guards."""
import collections, json, math, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_predictions
RHOS = (0, 0.16, 0.32, 0.48, 0.64, 0.8)
def score(pred, labels):
    labels = json.loads(Path(labels).read_text())
    if labels.get("format") != "dipmind-labels-v1": raise ValueError("unknown label schema")
    if any(labels.get(k) != v for k,v in {"n":100,"B":500,"nodes":200,"alpha":0.05}.items()): raise ValueError("scientific defaults changed")
    protocol = labels.get("protocol")
    if protocol == "full": expected_n = 200
    elif protocol == "smoke": expected_n = 1
    else: raise ValueError("unknown evaluation protocol")
    if labels.get("replications_per_cell") != expected_n: raise ValueError("invalid replication count")
    cases = labels["cases"]
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)) or not all(isinstance(x,str) and x for x in ids): raise ValueError("invalid case IDs")
    cells = collections.defaultdict(list)
    for c in cases:
        if type(c["model"]) is not int or c["model"] not in (1,2) or c["rho"] not in RHOS: raise ValueError("invalid model/rho")
        cells[(c["model"],c["rho"])].append(c["case_id"])
    expected = {(m,r) for m in (1,2) for r in RHOS}
    if set(cells) != expected or any(len(v) != expected_n for v in cells.values()): raise ValueError("full 12-cell grid with expected replications required")
    rows = read_predictions(pred)
    if set(rows) != set(ids): raise ValueError("predictions and labels must match exactly")
    power = {key:sum(rows[i] <= 0.05 for i in values)/len(values) for key,values in cells.items()}
    raw = math.fsum(power[(m,r)] for m in (1,2) for r in RHOS[1:])/10
    passed = power[(2,0)] <= 0.081 and power[(1,0)] <= 0.10
    return {"mean_power":raw if passed else 0.0,"unconstrained_mean_power":raw,
            "size_model1":power[(1,0)],"size_model2":power[(2,0)],"size_guards_pass":passed,
            "cells":[{"model":m,"rho":r,"n_replications":len(cells[(m,r)]),"rejection_rate":power[(m,r)]} for m,r in sorted(cells)]}
if __name__ == "__main__": print(json.dumps(score(*sys.argv[1:])))
