#!/usr/bin/env python3
"""Assemble results/metrics.json for the DiPMInd reproduction (stdlib only)."""
import csv
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
res = os.path.join(root, "results")

def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def load(path):
    with open(path) as fh:
        return {r["setting"]: {k: num(v) for k, v in r.items() if k != "setting"}
                for r in csv.DictReader(fh)}


rep, pap = load(os.path.join(res, "size_power_reproduced.csv")), \
           load(os.path.join(res, "size_power_paper.csv"))

V = ["jt", "jt-AD", "jt-F", "prod", "prod-AD", "prod-F"]
keys = sorted(rep)
nrep = None
with open(os.path.join(res, "all_pvalues.csv")) as fh:
    nrep = sum(1 for _ in csv.DictReader(fh)) // len(keys)
    fh.seek(0)
    rhos = sorted({float(r["rho"]) for r in csv.DictReader(fh)})

HL_KEY, HL_VAR = "model1_rho0.80", "prod-AD"
paper_v, rep_v = pap[HL_KEY][HL_VAR], rep[HL_KEY][HL_VAR]

diffs = [abs(rep[k][v] - pap[k][v]) for k in keys for v in V if rep[k][v] is not None]
m1 = [abs(rep[k][v] - pap[k][v]) for k in keys if k.startswith("model1_rho0.") and k != "model1_rho0.00" for v in V if rep[k][v] is not None]

metrics = {
    "status": ("reproduced" if rep_v is not None and abs(rep_v - paper_v) / paper_v <= 0.05
               else "metric_mismatch"),
    "main_experiment": ("Section 3, Example 1 / Figure 1: Monte-Carlo size and power of the six DiPMInd "
                        "joint independence tests between 200-node random networks (graph Laplacian matrices, "
                        "l1 metric) and Euclidean vectors (l1 metric), n = 100, alpha = 0.05, "
                        "B = 500 permutations, 200 Monte-Carlo runs per setting"),
    "headline_metric": {
        "name": ("empirical power (proportion of permutation p-values <= 0.05) of the prod-AD test, "
                 "Example 1 Model 1, rho = 0.8"),
        "higher_is_better": True,
        "setting": "n = 100, B = 500, 200 Monte-Carlo runs, alpha = 0.05",
        "paper_value": paper_v,
        "reproduced_value": rep_v,
        "relative_gap": (None if rep_v is None else (rep_v - paper_v) / paper_v),
        "paper_source": "Figure 1, left panel (Model 1), page 15; value read off the vector PDF of the arXiv source",
    },
    "secondary_metrics": {
        "size_rho0_mean_over_six_tests": {
            "model1": {"paper": sum(pap["model1_rho0.00"][v] for v in V) / 6,
                       "reproduced": sum(rep["model1_rho0.00"][v] for v in V) / 6},
            "model2": {"paper": sum(pap["model2_rho0.00"][v] for v in V) / 6,
                       "reproduced": sum(rep["model2_rho0.00"][v] for v in V) / 6}},
        "mean_abs_difference_all_table_entries": sum(diffs) / len(diffs),
        "mean_abs_difference_model1_nonnull": sum(m1) / len(m1),
    },
    "table_reproduced": rep,
    "table_paper": pap,
    "settings": {"n": 100, "B": 500, "alpha": 0.05, "monte_carlo_runs": nrep, "rho_grid": rhos,
                 "tests": V},
}

with open(os.path.join(res, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=1)
print("status:", metrics["status"],
      "| headline paper %.3f vs reproduced %.3f (rel gap %+.1f%%)"
      % (paper_v, rep_v, 100 * (rep_v - paper_v) / paper_v))
print("wrote results/metrics.json")
