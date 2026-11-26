"""Frozen structural checks. No NumPy or candidate imports."""
import json
import math

def load(path):
    def no_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def unique_object(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate object key: " + key)
            out[key] = value
        return out
    with open(path) as f:
        return json.load(f, parse_constant=no_constant, object_pairs_hook=unique_object)

def number(value, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite number")
    if nonnegative and value < 0 or positive and value <= 0:
        raise ValueError("invalid numeric range")
    return value

def rows(document):
    if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("cases"), list) or not document["cases"]:
        raise ValueError("expected version 1 with nonempty cases")
    out = {}
    for row in document["cases"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in out:
            raise ValueError("invalid/duplicate case id")
        out[row["id"]] = row
    return out

def predictions(prediction, expected):
    pred = rows(prediction)
    if set(pred) != set(expected):
        raise ValueError("case IDs differ from input/labels")
    for key, p in pred.items():
        if set(p) != {"id", "mean", "full_fit_evals", "wall_s", "reference_accesses"}:
            raise ValueError("unexpected prediction fields")
        d = expected[key]["d"]
        if not isinstance(p["mean"], list) or len(p["mean"]) != d:
            raise ValueError("posterior mean dimension mismatch")
        for v in p["mean"]:
            number(v)
        for name in ("full_fit_evals", "reference_accesses"):
            if isinstance(p[name], bool) or not isinstance(p[name], int) or p[name] < 0:
                raise ValueError("expected nonnegative integer telemetry")
        number(p["wall_s"], positive=True)
    return pred
