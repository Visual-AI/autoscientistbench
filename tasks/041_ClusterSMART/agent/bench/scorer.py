#!/usr/bin/env python3
"""Mean power with per-environment coverage guard; all labels remain scorer-side."""
import csv,json,math,sys
from collections import defaultdict

def rows(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))
def score(pred,labels):
    data=rows(pred); truth=rows(labels)
    by_id={r["id"]:r for r in data}
    if len(by_id)!=len(data) or set(by_id)!={r["id"] for r in truth}:raise ValueError("prediction IDs disagree with labels")
    groups=defaultdict(list)
    for t in truth:
        r=by_id[t["id"]]; est,lo,hi=(float(r[k]) for k in ("estimate","lower","upper"))
        if not all(map(math.isfinite,(est,lo,hi))) or not lo<=est<=hi:raise ValueError("invalid interval")
        v=float(t["true_value"])
        groups[t["group"]].append((int(lo>0 or hi<0),int(lo<v<hi),float(t["coverage_min"])))
    if set(groups)!={"g1","g2","g3","g4"}:raise ValueError("all four environments required")
    detail={k:{"n":len(v),"power":sum(x[0] for x in v)/len(v),"coverage":sum(x[1] for x in v)/len(v),"coverage_min":v[0][2]} for k,v in groups.items()}
    good=all(d["coverage"]>=d["coverage_min"] for d in detail.values())
    power=sum(d["power"] for d in detail.values())/4
    return {"power":power if good else 0.0,"unconstrained_power":power,"coverage_guard_pass":good,"environments":detail}
if __name__=="__main__":print(json.dumps(score(sys.argv[1],sys.argv[2]),allow_nan=False))
