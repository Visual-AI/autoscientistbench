#!/usr/bin/env python3
"""Mean positive-rho setting RMSE, guarded by coverage over ALL settings."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import read_predictions, expected_keys

def score(pred, labels):
    with open(labels) as handle:
        target = json.load(handle)
    settings = target["settings"]
    if target["protocol"] == "full":
        grid = {(d, rho, n) for d in (2,4,6,8,10) for rho in (0,0.3,0.5,0.7,0.9) for n in (300,600,900)}
        actual = {(s["d"], s["rho"], s["n"]) for s in settings}
        if actual != grid or len(settings) != 75 or any(s["replications"] != 1000 for s in settings):
            raise ValueError("full scoring requires all 75 settings and 1000 replications")
    elif target["protocol"] != "smoke":
        raise ValueError("unknown protocol")
    rows = read_predictions(pred)
    if set(rows) != expected_keys(settings):
        raise ValueError("prediction/label IDs must match exactly")
    diagnostics = []
    for setting in settings:
        truth = float(setting["truth"])
        if not math.isfinite(truth):
            raise ValueError("nonfinite truth")
        observations = [rows[(setting["setting_id"], i)] for i in range(1, setting["replications"]+1)]
        mse = math.fsum((estimate-truth)**2 for estimate, lower, upper in observations)/len(observations)
        ecp = sum(lower <= truth <= upper for estimate, lower, upper in observations)/len(observations)
        diagnostics.append({"setting_id": setting["setting_id"], "rho": setting["rho"], "rmse": math.sqrt(mse), "ecp": ecp})
    positive = [item["rmse"] for item in diagnostics if item["rho"] > 0]
    if not positive:
        raise ValueError("no positive-rho settings")
    rmse = math.fsum(positive)/len(positive)
    mean_ecp = math.fsum(item["ecp"] for item in diagnostics)/len(diagnostics)
    min_ecp = min(item["ecp"] for item in diagnostics)
    coverage_pass = mean_ecp >= 0.94 and min_ecp >= 0.85
    return {"mean_rmse": rmse if coverage_pass else float("inf"), "unconstrained_mean_rmse": rmse,
            "mean_ecp": mean_ecp, "min_ecp": min_ecp, "coverage_pass": coverage_pass,
            "settings": len(settings), "positive_rho_settings": len(positive), "details": diagnostics}

if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:])))
