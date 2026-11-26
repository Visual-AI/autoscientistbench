#!/usr/bin/env python3
"""Section 4.2: mean_n mean_r median_j standardized absolute mean bias.
Cap 21500 is the explicit reviewed correction in tasks_merlin.json.review.
Telemetry is candidate-produced: checking its truth requires patch audit.
"""
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, rows, predictions, number

EVALUATION_CAP = 21500

def aggregate(values, expected):
    groups = {}
    for key, value in values.items():
        groups.setdefault(expected[key]["n"], []).append(value)
    return statistics.mean(statistics.mean(v) for v in groups.values())

def score(prediction, labels):
    expected = rows(labels)
    fallback = {}
    for key, row in expected.items():
        d = row["d"]
        if len(row["reference_mean"]) != d or len(row["reference_sd"]) != d:
            raise ValueError("invalid reference dimensions")
        for value in row["reference_mean"]:
            number(value)
        for value in row["reference_sd"]:
            number(value, positive=True)
        fallback[key] = number(row["baseline_lap_bias"], nonnegative=True)
        number(row["baseline_slap_wall_s"], positive=True)
    worst = aggregate(fallback, expected)
    violations = []
    biases = {}
    ratios = []
    counts = []
    try:
        pred = predictions(prediction, expected)
        for key, p in pred.items():
            row = expected[key]
            bias = statistics.median(abs(a-b)/s for a,b,s in zip(p["mean"],row["reference_mean"],row["reference_sd"]))
            number(bias, nonnegative=True)
            biases[key] = bias
            counts.append(p["full_fit_evals"])
            ratio = p["wall_s"] / row["baseline_slap_wall_s"]
            ratios.append(ratio)
            if p["full_fit_evals"] > EVALUATION_CAP:
                violations.append(key + ": evaluations exceed 21500")
            if p["reference_accesses"] != 0:
                violations.append(key + ": reference accessed")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        violations.append("invalid prediction: " + str(exc))
    raw = aggregate(biases, expected) if len(biases) == len(expected) else None
    return {"standardized_bias": worst if violations else raw,
            "raw_standardized_bias": raw, "plain_laplace_fallback": worst,
            "guards_passed": not violations, "violations": violations,
            "max_full_fit_evals": max(counts, default=None),
            "max_wall_ratio": max(ratios, default=None), "cases": len(expected),
            "telemetry_provenance": "patch_audit_required"}

def main():
    labels = load(sys.argv[2])
    try:
        pred = load(sys.argv[1])
    except (ValueError, OSError) as exc:
        pred = None
    print(json.dumps(score(pred, labels), allow_nan=False))

if __name__ == "__main__":
    main()
