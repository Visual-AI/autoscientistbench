"""DESIGN §6: gain / threshold / PASS / aggregation. Shared by judge and eval-dev."""
import math
import statistics

FAIL_GAIN = -0.10


def aggregate(values):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"per_seed": list(values), "mean": None, "std": None, "n": 0}
    nonfinite = sum(1 for v in vals if not math.isfinite(v))
    if nonfinite:
        # a scorer that prints +/-inf means a guard was violated: no mean, decide() -> gain None -> FAIL_GAIN, not PASS
        return {"per_seed": [v if math.isfinite(v) else None for v in vals], "mean": None, "std": None, "n": len(vals),
                "nonfinite": nonfinite}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {"per_seed": vals, "mean": mean, "std": std, "n": len(vals)}


def gain(cand_mean, ref_mean, higher_is_better):
    if cand_mean is None or ref_mean is None:
        return None
    sign = 1.0 if higher_is_better else -1.0
    denom = abs(ref_mean) if abs(ref_mean) > 1e-12 else 1e-12
    return sign * (cand_mean - ref_mean) / denom


def threshold(delta, ref_std, ref_mean):
    denom = abs(ref_mean) if ref_mean is not None and abs(ref_mean) > 1e-12 else 1e-12
    noise = 2.0 * (ref_std or 0.0) / denom
    return max(float(delta), noise)


def decide(cand_mean, ref_mean, ref_std, higher_is_better, delta):
    g = gain(cand_mean, ref_mean, higher_is_better)
    thr = threshold(delta, ref_std, ref_mean)
    if g is None:
        return {"gain": None, "gain_clipped": FAIL_GAIN, "threshold": thr, "pass": False}
    return {"gain": g, "gain_clipped": max(g, FAIL_GAIN), "threshold": thr, "pass": bool(g >= thr)}


def parity_ok(observed, archived_mean, archived_std, rel_tol=0.0, abs_tol=0.0):
    """Reference re-run must land within sigma/2 (+ optional extras) of the archived mean."""
    if observed is None or archived_mean is None:
        return False, None
    tol = 0.5 * (archived_std or 0.0) + abs(archived_mean) * float(rel_tol) + float(abs_tol)
    diff = abs(observed - archived_mean)
    return diff <= tol + 1e-12, {"diff": diff, "tol": tol}
