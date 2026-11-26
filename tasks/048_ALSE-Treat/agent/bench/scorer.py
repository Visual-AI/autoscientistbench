#!/usr/bin/env python3
"""Manifest scaled MSE with pooled and every-cell Wald coverage guards."""
import collections
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import expected,read_predictions

def score(pred,labels):
    target=json.loads(Path(labels).read_text())
    cases=target["cases"]
    keys=expected(cases)
    counts=collections.Counter(c["n"] for c in cases)
    if target["protocol"]=="full":
        if counts!=dict.fromkeys(range(500,4001,500),40):
            raise ValueError("full scoring requires 40 datasets at every n=500,1000,...,4000")
    elif target["protocol"]!="smoke":raise ValueError("unknown scoring protocol")
    rows=read_predictions(pred)
    if set(rows)!=keys:raise ValueError("predictions and labels must match exactly")
    cells=collections.defaultdict(list)
    for case in cases:
        truth=float(case["truth"])
        if not math.isfinite(truth):raise ValueError("invalid truth")
        for learner in ("gam","ranger"):
            estimate,se=rows[(case["case_id"],case["n"],learner)]
            cells[(case["n"],learner)].append((case["n"]*(estimate-truth)**2,abs(estimate-truth)<=1.959964*se))
    details=[]
    for (n,learner),values in sorted(cells.items()):
        details.append({"n":n,"learner":learner,"datasets":len(values),
                        "scaled_mse":math.fsum(v[0] for v in values)/len(values),
                        "coverage":sum(v[1] for v in values)/len(values)})
    unconstrained=math.fsum(c["scaled_mse"] for c in details)/len(details)
    pooled=sum(v[1] for values in cells.values() for v in values)/len(rows)
    minimum=min(c["coverage"] for c in details)
    passed=pooled>=0.92 and minimum>=0.85
    return {"scaled_mse":unconstrained if passed else float("inf"),"unconstrained_scaled_mse":unconstrained,
            "pooled_coverage":pooled,"minimum_cell_coverage":minimum,"coverage_pass":passed,"cells":details}
if __name__=="__main__":print(json.dumps(score(*sys.argv[1:])))
