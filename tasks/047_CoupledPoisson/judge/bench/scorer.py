#!/usr/bin/env python3
"""Manifest: sample variance * average counted MCMC transitions, with 3-SE guard."""
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import read_predictions

def score(pred, labels):
    try:
        rows = read_predictions(pred)
        truth = json.loads(Path(labels).read_text())
        n = len(rows)
        if truth["scope"] not in ("full", "smoke"):
            raise ValueError("unknown evaluation scope")
        required = 1000 if truth["scope"] == "full" else truth["n"]
        if n != required or truth["n"] != required:
            raise ValueError("wrong independent-run count")
        if len(truth["case_ids"]) != n or len(set(truth["case_ids"])) != n or set(truth["case_ids"]) != {r["case_id"] for r in rows}:
            raise ValueError("truth/prediction IDs differ")
        target = float(truth["asymptotic_variance"])
        if target != 10000.0:
            raise ValueError("unexpected frozen AR(1) estimand")
        estimates = [r["estimate"] for r in rows]
        mean = statistics.fmean(estimates)
        variance = statistics.variance(estimates)
        average_cost = statistics.fmean(r["counted_cost"] for r in rows)
        bound = 3 * math.sqrt(variance / n)
        guard = abs(mean - target) <= bound
        return {"asymptotic_inefficiency": variance * average_cost if guard else float("inf"),
                "mean": mean, "sample_variance": variance, "average_counted_cost": average_cost,
                "bias_guard_pass": guard, "three_standard_errors": bound, "n": n,
                "scope": truth["scope"], "cost_arithmetic_checked": True,
                "cost_interface_requires_patch_audit": True}
    except Exception as exc:
        return {"asymptotic_inefficiency": float("inf"), "guard_failure": str(exc)}

if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:])))
