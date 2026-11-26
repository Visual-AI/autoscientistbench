#!/usr/bin/env python3
"""Bench entry points (protocol only; the method is the authors' code under src/ and is unchanged).

  train   --out DIR                       writes DIR/config.json (the paper's Table-1 hyper-parameters)
  predict --model DIR --input F --output G  applies the method to every instance of F -> CSV rows id,w_star,L11,L21,L22,eps

Per instance (exactly Sec. 6.2 / App. E.2 at J = 10, as in the archived reproduction):
  reference  P_hat  = empirical distribution of the 10 observed points (x_j, y_j)
  bootstrap  P_k    = empirical distributions of the 20 given resamples
  eps_0             = 0.9-quantile (gamma = 0.1) of the type-1 Wasserstein distances d(P_hat, P_k; L0), L0 = I
  L*                = authors' bilevel optimisation (Adam, lr 1e-4, lambda 10, eta 100, rel_impr_obj 1e-6, <= 1e6 it.)
  w*                = DRO decision at (L*, eps_0), i.e. the last stored decision of the optimiser
The method has no random component: BENCH_SEED is honoured (numpy seeded) but does not change the result.
Instances run in a process pool of BENCH_CPUS workers, one BLAS thread each.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = {
    "wasserstein_type": 1,
    "gamma": 0.1,
    "L0": [[1.0, 0.0], [0.0, 1.0]],
    "opt_params": {
        "n_iter_max": 1000000, "learning_rate": 1e-4, "store_every": 100,
        "stopping_tols": {"diff_decision": 0.0, "diff_objective": 0.0, "diff_objective_pen": 0.0, "diff_weightmat": 0.0,
                          "weightmat_gradient": 0.0, "rel_impr_obj": 1e-6, "rel_diff_obj_pen": 0.0},
        "penalization": {"lambda": 10.0, "eta": 100.0},
    },
}


def empirical(samples):
    """Discrete distribution (unique support, empirical probabilities) of an (n, 2) sample — as in the authors' examples."""
    supp = np.unique(samples, axis=0)
    prob = np.array([np.sum(np.all(samples == s, axis=1)) / len(samples) for s in supp])
    return supp, prob


def solve_instance(args):
    iid, x, y, idx, cfg = args
    from trainable_ot_dro.bilevel_optimization import optimize_transportation_matrix
    from trainable_ot_dro.reformulations import reform_linreg_dro
    from trainable_ot_dro.utils.wasserstein_distance import wasserstein_distance_L, wasserstein_gradient_L
    t0 = time.perf_counter()
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    samples = np.hstack((X, y.reshape(-1, 1)))
    ref = empirical(samples)
    boots = [empirical(samples[i]) for i in np.asarray(idx)]
    L0 = np.array(cfg["L0"], dtype=float)
    wtype = int(cfg["wasserstein_type"])
    distances = np.array([wasserstein_distance_L(ref, b, L0, wtype)[0] for b in boots])
    eps = float(np.quantile(distances, 1.0 - cfg["gamma"]))
    cp = reform_linreg_dro(X, y, eps, L0, wasserstein_type=wtype)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = optimize_transportation_matrix(L0, cp, {"reference": ref, "bootstrap": boots},
                                                {"gradient": wasserstein_gradient_L, "type": wtype},
                                                {"gamma": cfg["gamma"], "eps": eps}, cfg["opt_params"])
    L = np.array(result["L_matrices"][-1], dtype=float)
    w = float(np.asarray(result["decisions"][-1]).ravel()[0])
    return {"id": iid, "w_star": w, "L11": float(L[0, 0]), "L21": float(L[1, 0]), "L22": float(L[1, 1]), "eps": eps,
            "iterations": int(result["iterations"][-1]) + 1, "f0": float(result["objective_values"][0]),
            "f_star": float(result["objective_values"][-1]), "stop": str(result["stopping_reasons"]).strip(),
            "wall_s": time.perf_counter() - t0}


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG, seed=int(os.environ.get("BENCH_SEED", "0")))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"train: nothing to fit (L is learned per instance at prediction time); wrote {out / 'config.json'}")


def cmd_predict(a):
    seed = int(os.environ.get("BENCH_SEED", "0"))
    np.random.seed(seed)                      # the method is deterministic; seeded for protocol compliance
    cfg_path = Path(a.model) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else dict(DEFAULT_CONFIG)
    with np.load(a.input, allow_pickle=False) as z:
        ids = [str(i) for i in z["id"]]
        xs, ys, idxs = z["x"], z["y"], z["bootstrap_indices"]
    workers = max(1, min(int(os.environ.get("BENCH_CPUS", "1")), len(ids)))
    jobs = [(ids[i], xs[i], ys[i], idxs[i], cfg) for i in range(len(ids))]
    print(f"predict: {len(ids)} instances, {workers} workers, seed {seed}", flush=True)
    t0 = time.perf_counter()
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(solve_instance, jobs):
            rows.append(r)
            print(json.dumps({"done": len(rows), "total": len(ids), "elapsed_s": round(time.perf_counter() - t0, 1),
                              **{k: r[k] for k in ("id", "w_star", "eps", "iterations", "f0", "f_star", "wall_s")}}), flush=True)
    rows.sort(key=lambda r: ids.index(r["id"]))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write("id,w_star,L11,L21,L22,eps\n")
        for r in rows:
            f.write(",".join([r["id"]] + [repr(float(r[k])) for k in ("w_star", "L11", "L21", "L22", "eps")]) + "\n")
    tmp.replace(out)
    print(f"predict: wrote {out} ({len(rows)} rows) in {time.perf_counter() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("train"); p.add_argument("--out", required=True); p.set_defaults(f=cmd_train)
    p = sp.add_parser("predict"); p.add_argument("--model", required=True); p.add_argument("--input", required=True)
    p.add_argument("--output", required=True); p.set_defaults(f=cmd_predict)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
