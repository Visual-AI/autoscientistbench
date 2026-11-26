"""Reconstructed train-only bounded Powell fit; no author optimizer was released."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, os.environ["BENCH_BENCH_DIR"])
from scipy.optimize import minimize
from expressions import Model, physical_checks, read_table
from baseline import BOUNDS, INITIAL, PARAMETER_NAMES, certificate

def fit(rows, maxfev=4000):
    model = Model(certificate())
    best = {"loss": float("inf"), "parameters": None}
    calls = 0
    def objective(parameters):
        nonlocal calls
        calls += 1
        model.parameters = [float(p) for p in parameters]
        try:
            physical_checks(model)
            errors = [math.log(model.predict(row)["xi"] / row["xi"]) ** 2 for row in rows]
            loss = math.fsum(errors) / len(errors)
            if not math.isfinite(loss):
                raise ValueError("invalid objective")
        except (ValueError, OverflowError, ZeroDivisionError):
            return 1e12
        if loss < best["loss"]:
            best.update(loss=loss, parameters=list(model.parameters))
        return loss
    initial_loss = objective(INITIAL)
    if best["parameters"] is None:
        raise RuntimeError("neutral initialization fails the physical checks")
    result = minimize(objective, INITIAL, method="Powell", bounds=BOUNDS,
                      options=dict(maxfev=maxfev, maxiter=1000, xtol=1e-4, ftol=1e-7))
    return certificate(best["parameters"]), dict(
        solver="scipy.optimize.minimize/Powell", reconstructed_fit=True,
        initial_log_mse=initial_loss, train_log_mse=best["loss"],
        objective_calls=calls, optimizer_success=bool(result.success),
        optimizer_message=str(result.message), global_parameter_names=PARAMETER_NAMES,
        maxfev=maxfev, physical_checks="enforced for every objective evaluation",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train_path = Path(os.environ["BENCH_DATA_DIR"]) / "train/train.csv"
    rows = read_table(train_path, labels=True)
    maxfev = int(os.environ.get("PHY13_FIT_MAXFEV", "4000"))
    if maxfev < 1:
        raise ValueError("PHY13_FIT_MAXFEV must be positive")
    cert, report = fit(rows, maxfev=maxfev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(cert, indent=2, allow_nan=False) + "\n")
    report.update(seed=int(os.environ.get("BENCH_SEED", "1")), deterministic=True,
                  training_rows=len(rows), training_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest())
    (out / "fit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fitted_global_scalars": len(cert["parameters"]), "training_rows": len(rows), "objective_calls": report["objective_calls"]}))

if __name__ == "__main__":
    main()
