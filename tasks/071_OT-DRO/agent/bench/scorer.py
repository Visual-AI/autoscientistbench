#!/usr/bin/env python3
"""Task metric and guards (the manifest's score_rule, review-revised), computed on the instances of the labels file.

    score = mean over instances of  e_oos(w) = E_{x~U(-10,10), e~N(0,sigma_m^2)} |(w_m - w) x + e|      (lower is better)
            if  (A) every submitted decision w equals the DRO optimum w*(L, eps) of its own submitted (L, eps),
                (B) mean bootstrap exceedance  <= max(0.10, baseline) + 0.05,
                (C) true-P coverage rate       >= baseline - 0.05;
            otherwise +inf.

e_oos is the exact 1-D expectation: for fixed x the normal absolute moment is closed-form, the x-integral is
2001-node Gauss-Legendre quadrature (no Monte-Carlo noise).  "baseline" is the unchanged authors' method evaluated
on the *same* instances (its per-instance exceedance / coverage are frozen in the labels file); on the paper's 100
archived instances this gives exactly the review's thresholds 0.40 and 0.26.

(A) w*(L, eps) = argmin_w  mean_j |y_j - w x_j| + eps * || L^{-1} (-w, 1) ||_2   (type-1 Wasserstein reformulation,
    Chen & Paschalidis eq. 4.5, the same objective the authors' conic program minimises).  The scorer solves this
    strictly convex 1-D problem by bisection on the subgradient; the submission passes if |w - w*| <= 1e-3 max(1, |w*|)
    or if its objective is within 1e-8 (relative) of the optimum.  (Clarabel's own solutions sit within 1.6e-4 / 2.6e-7.)
(B) exceedance of one instance = fraction of its 20 frozen bootstrap resamples P_k with d(P_hat, P_k; L) > eps,
    d = type-1 Wasserstein distance with ground cost ||L^T (xi_1 - xi_2)||_2 (POT exact EMD).
(C) an instance is covered if d(P_hat, P_M; L) <= eps, P_M = the frozen M = 5000-point sample of the true P
    (generator.true_sample with the instance's coverage_seed).

Usage: scorer.py <pred.csv> <truth.npz>   -> prints one JSON line {"e_oos": value | Infinity, ...diagnostics}
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generator  # noqa: E402  (frozen DGP, same directory)
import ot  # noqa: E402
from scipy.special import ndtr, roots_legendre  # noqa: E402

EXC_FLOOR, EXC_SLACK, COV_SLACK = 0.10, 0.05, 0.05       # guards (B), (C)
ABS_EXC_MAX, ABS_COV_MIN = 0.40, 0.26                     # fallback if the labels carry no baseline (= review's numbers)
W_TOL, OBJ_TOL = 1e-3, 1e-8                               # guard (A)
NODES, WEIGHTS = roots_legendre(2001)
COLS = ("w_star", "L11", "L21", "L22", "eps")


def e_oos(w_true, w, sigma):
    """E |(w_true - w) x + e|, x ~ U(-10, 10), e ~ N(0, sigma^2); exact in e, Gauss-Legendre in x."""
    a = (w_true - w) * 10.0 * NODES
    expectation = sigma * np.sqrt(2.0 / np.pi) * np.exp(-a * a / (2.0 * sigma * sigma)) + a * (1.0 - 2.0 * ndtr(-a / sigma))
    return float(np.dot(WEIGHTS, expectation) / 2.0)


def empirical(samples):
    supp = np.unique(samples, axis=0)
    prob = np.array([np.sum(np.all(samples == s, axis=1)) / len(samples) for s in supp])
    return supp, prob


def w1_distance(d1, d2, L):
    """Type-1 Wasserstein distance between discrete distributions for the cost ||L^T (xi_1 - xi_2)||_2."""
    (s1, p1), (s2, p2) = d1, d2
    diff = s1[:, None, :] - s2[None, :, :]
    cost = np.sqrt(np.sum((diff @ L) ** 2, axis=-1))
    plan = ot.emd(p1, p2, cost)
    return float(np.sum(plan * cost))


def objective(w, x, y, eps, L):
    return float(np.mean(np.abs(y - w * x)) + eps * np.linalg.norm(np.linalg.solve(L, np.array([-w, 1.0]))))


def subgradient(w, x, y, eps, L):
    r = y - w * x
    g = -np.mean(x * np.sign(r))
    v = np.linalg.solve(L, np.array([-w, 1.0]))
    dv = np.linalg.solve(L, np.array([-1.0, 0.0]))
    return g + eps * float(v @ dv) / float(np.linalg.norm(v))


def dro_optimum(x, y, eps, L, lo=-1e6, hi=1e6):
    """argmin of the strictly convex objective by bisection on its (monotone) subgradient."""
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if subgradient(mid, x, y, eps, L) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def read_pred(path):
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["id"]] = {c: float(row[c]) for c in COLS}
    return out


def main(pred_path, labels_path):
    pred = read_pred(pred_path)
    with np.load(labels_path, allow_pickle=False) as z:
        lab = {k: z[k] for k in z.files}
    ids = [str(i) for i in lab["id"]]
    n = len(ids)
    per = []
    for i, iid in enumerate(ids):
        if iid not in pred:
            sys.exit(f"missing prediction for id {iid}")
        p = pred[iid]
        x, y, idx = lab["x"][i], lab["y"][i], lab["bootstrap_indices"][i]
        w_true, sigma = float(lab["w_true"][i]), float(np.sqrt(lab["sigma2"][i]))
        w, eps = p["w_star"], p["eps"]
        L = np.array([[p["L11"], 0.0], [p["L21"], p["L22"]]])
        rec = {"id": iid, "e_oos": e_oos(w_true, w, sigma), "w": w, "eps": eps}
        finite = all(math.isfinite(v) for v in p.values())
        if not finite or eps <= 0 or p["L11"] <= 0 or p["L22"] <= 0:
            rec.update(decision_ok=False, exceedance=1.0, covered=0, reason="eps<=0 or L not a positive lower-triangular factor")
            per.append(rec)
            continue
        w_opt = dro_optimum(x, y, eps, L)
        j_opt, j_sub = objective(w_opt, x, y, eps, L), objective(w, x, y, eps, L)
        dw, gap = abs(w - w_opt), j_sub - j_opt
        rec.update(w_opt=w_opt, dw=dw, obj_gap=gap,
                   decision_ok=bool(dw <= W_TOL * max(1.0, abs(w_opt)) or gap <= OBJ_TOL * max(1.0, abs(j_opt))))
        samples = np.column_stack((x, y))
        ref = empirical(samples)
        dists = np.array([w1_distance(ref, empirical(samples[k]), L) for k in idx])
        rec["exceedance"] = float(np.mean(dists > eps))
        true_p = generator.true_sample(int(lab["model_id"][i]), int(lab["coverage_seed"][i]))
        rec["true_distance"] = w1_distance(ref, true_p, L)
        rec["covered"] = int(rec["true_distance"] <= eps)
        per.append(rec)
    mean_e = float(np.mean([r["e_oos"] for r in per]))
    n_bad = sum(1 for r in per if not r["decision_ok"])
    exc = float(np.mean([r["exceedance"] for r in per]))
    cov = float(np.mean([r["covered"] for r in per]))
    base_exc = lab.get("baseline_exceedance")
    base_cov = lab.get("baseline_covered")
    if base_exc is not None and base_cov is not None and np.all(np.isfinite(base_exc)) and np.all(np.isfinite(base_cov)):
        b_exc, b_cov = float(np.mean(base_exc)), float(np.mean(base_cov))
        exc_max, cov_min, src = max(EXC_FLOOR, b_exc) + EXC_SLACK, b_cov - COV_SLACK, "baseline on these instances"
    else:
        b_exc = b_cov = None
        exc_max, cov_min, src = ABS_EXC_MAX, ABS_COV_MIN, "absolute (review)"
    ok_b, ok_c = exc <= exc_max + 1e-9, cov >= cov_min - 1e-9
    guards_ok = n_bad == 0 and ok_b and ok_c
    out = {
        "e_oos": mean_e if guards_ok else float("inf"),
        "e_oos_raw": mean_e, "n_instances": n, "guards_ok": guards_ok,
        "decision_violations": n_bad, "max_decision_dw": max((r.get("dw", float("inf")) for r in per), default=None),
        "bootstrap_exceedance": exc, "bootstrap_exceedance_max": exc_max, "exceedance_ok": ok_b,
        "coverage_rate": cov, "coverage_min": cov_min, "coverage_ok": ok_c,
        "baseline_exceedance": b_exc, "baseline_coverage_rate": b_cov, "thresholds_from": src,
        "noise_floor": float(np.mean([generator.noise_floor(int(m)) for m in lab["model_id"]])),
    }
    if "baseline_e_oos" in lab and np.all(np.isfinite(lab["baseline_e_oos"])):
        out["baseline_e_oos"] = float(np.mean(lab["baseline_e_oos"]))
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: scorer.py <pred.csv> <truth.npz>")
    main(sys.argv[1], sys.argv[2])
