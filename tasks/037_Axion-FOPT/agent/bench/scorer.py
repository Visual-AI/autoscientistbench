#!/usr/bin/env python3
"""Recompute an inert <=7-scalar model and all four original physical checks."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expressions import Model, load, physical_checks, read_table, validate_prediction

GUARD_CODES = {
    "predictions differ from certified model": "prediction_certificate_mismatch",
    "physical smoke: sensible result": "physical_sensible_result",
    "physical smoke: high-H floor and limit": "physical_high_h_limit",
    "physical smoke: monotonic H decline and enhancement": "physical_h_monotonicity",
    "physical smoke: low-H slope": "physical_low_h_slope",
}

def score(pred_path, labels_path):
    try:
        labels = read_table(labels_path, labels=True)
        prediction = load(pred_path)
        claimed = validate_prediction(prediction, labels)
        model = Model(prediction["model"])
        squared = []
        for row, supplied in zip(labels, claimed):
            xi = model.predict(row)["xi"]
            if not math.isclose(xi, supplied["xi"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("predictions differ from certified model")
            squared.append(math.log(xi / row["xi"]) ** 2)
        checks = physical_checks(model)
        metric = math.sqrt(math.fsum(squared) / len(squared))
        return dict(log_rmse=metric, raw_score=metric, global_fitted_scalars=len(model.parameters),
                    physical_checks=checks, certificate_verified=True)
    except Exception as exc:
        return dict(log_rmse=float("inf"), raw_score=1.,
                    guard_failure=GUARD_CODES.get(str(exc), "invalid_input_or_certificate"))


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2])))
