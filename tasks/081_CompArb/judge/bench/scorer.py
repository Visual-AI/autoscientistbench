#!/usr/bin/env python3
"""scorer.py PRED LABELS: USD buy cost to reach the target solve rate, scaled to the 500-issue benchmark. Lower is better.

Outcomes and prices come only from LABELS. PRED supplies plans and nothing else: its one numeric column besides the level
index is the dollars inside a plan. A malformed family or one that cannot reach the target scores Infinity.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import FamilyError, envelope_cost, family_points, load_family, load_labels, plain_interpolated_cost  # noqa: E402

METRIC = "buy_cost_usd"


def score(pred_path, labels_path):
    target, size, curves = load_labels(labels_path)
    try:
        family = load_family(pred_path, list(curves))
    except FamilyError as exc:
        return {METRIC: math.inf, "reason": str(exc)[:500]}
    solve, cost = family_points(family, curves)
    best, mix = envelope_cost(solve, cost, target)
    result = {
        METRIC: best * size if math.isfinite(best) else math.inf,
        "target_solve_rate": target,
        "frontier_solve_rate": max(solve),
        "n_levels": len(family),
        "n_issues": len(curves),
        "non_envelope_usd": plain_interpolated_cost(solve, cost, target) * size,
    }
    if not math.isfinite(best):
        result["reason"] = "the family does not reach the target solve rate"
    else:
        result["mix"] = {"low_level": mix[0], "high_level": mix[1], "high_weight": mix[2]}
    return result


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2])))
