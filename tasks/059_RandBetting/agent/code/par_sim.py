#!/usr/bin/env python3
"""Parallel driver for the four main-text simulation studies of

    Ramdas & Manole, "Randomized and exchangeable improvements of Markov's,
    Chebyshev's and Chernoff's inequalities", arXiv:2304.02611v3
    (Section 9 "Experiments", Figures 1-5).  In the older numbering used by
    the repository README these are Sections 7.1-7.4.

ADDED BY THIS REPRODUCTION.  The authors' scripts (`intervals.py`,
`Ke-vals.py`, `betting.py`, `mixture_ui.py`) are left byte-identical.  This
driver re-hosts their per-replication kernels inside a multiprocessing pool
because the seeding scheme of every script makes the outer-loop units (sample
size i / model j / replication index) independent: each unit starts by calling
`np.random.seed(c)` for a counter c that is a function of that unit only, and
never depends on the RNG state left behind by other units.  Hence every worker
reproduces exactly the same random stream as the serial script and the
assembled matrices are bit-identical to a serial run (checked against the
unmodified scripts on a reduced configuration, see REPRODUCTION.md).

Usage:
  python par_sim.py {intervals,kevals,betting,ui} --outdir DIR [--workers 12]
                    [--reps R] [--n-ns K] [--B B] [--n-starts S]
Writes the same .npy files, in the same shapes and with the same names, as the
corresponding authors' script (relative to --outdir).
"""
import argparse
import importlib.util
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {}


# --------------------------------------------------------------------------- #
# defs loading                                                            #
# --------------------------------------------------------------------------- #
def load_defs():
    """Import the authors' defs.py without changing sys.path."""
    spec = importlib.util.spec_from_file_location("defs_authors", os.path.join(HERE, "defs.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["defs_authors"] = mod  # fmix/confseq are not needed here
    spec.loader.exec_module(mod)
    return mod


def make_cfg(args):
    d = load_defs()
    reps = args.reps if args.reps is not None else d.reps
    n_ns = args.n_ns if args.n_ns is not None else d.n_ns
    ns = np.linspace(100, 2000, n_ns, dtype=int) if args.n_ns is not None else d.ns
    return dict(
        alpha=d.alpha,
        reps=int(reps),
        n_ns=int(n_ns),
        n_mods=d.n_mods,
        ns=np.asarray(ns),
        B=int(args.B if args.B is not None else d.B),
        M=d.M,
        R=d.R,
        mus=d.mus,
        rhos=d.rhos,
        a=d.a,
        b=d.b,
        interval_reps=int(args.reps if args.reps is not None else d.interval_reps),
        e_max=d.e_max,
        eps=d.eps,
        pi0=d.pi0,
        v0=d.v0,
        n_starts=int(args.n_starts if args.n_starts is not None else d.n_starts),
    )


def _init(cfg):
    global CFG
    CFG = cfg
    # keep BLAS from oversubscribing the 12 allocated cores
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(max(1, int(os.environ.get("REPRO_THREADS", "12")))))


# --------------------------------------------------------------------------- #
# Section 9.1 -- intervals.py  (unit = i, the sample size index)          #
# --------------------------------------------------------------------------- #
def intervals_unit(i):
    from scipy.stats import norm

    alpha, reps = CFG["alpha"], CFG["interval_reps"]
    n = int(CFG["ns"][i])
    zalpha = norm.ppf(1 - alpha / 2.0)
    c = {k: np.zeros(reps) for k in ("chb", "hof", "umi", "opt")}
    l = {k: np.zeros(reps) for k in ("chb", "hof", "umi", "opt")}
    ctr = i * reps
    for rep in range(reps):
        np.random.seed(ctr)
        x = np.random.normal(0, 1, n)
        xbar = np.sum(x) / n
        np.random.seed(ctr)
        u = np.random.uniform(0, 1, 1)

        bdry_opt = zalpha / np.sqrt(n)
        bdry_chb = 1 / np.sqrt(alpha * n)
        bdry_hof = np.sqrt(2 * np.log(2.0 / alpha) / n)
        bdry_umi = (np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
                    + np.log(2.0 * u / alpha) / np.sqrt(2.0 * n * np.log(2.0 / alpha)))

        for k, bd in (("opt", bdry_opt), ("chb", bdry_chb), ("hof", bdry_hof), ("umi", bdry_umi)):
            l[k][rep] += 2 * bd
            c[k][rep] += np.abs(xbar) <= bd
        ctr += 1
    return i, c, l


def run_intervals(outdir, workers, chunk):
    cfg = CFG
    out = os.path.join(outdir, "matrices", "intervals")
    os.makedirs(out, exist_ok=True)
    c = {k: np.zeros((cfg["n_ns"], cfg["interval_reps"])) for k in ("chb", "hof", "umi", "opt")}
    l = {k: np.zeros((cfg["n_ns"], cfg["interval_reps"])) for k in ("chb", "hof", "umi", "opt")}
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, ci, li in pool.imap_unordered(intervals_unit, range(cfg["n_ns"]), chunksize=1):
            for k in c:
                c[k][i] = ci[k]
                l[k][i] = li[k]
    for k in c:
        np.save(os.path.join(out, f"c_{k}.npy"), c[k])
        np.save(os.path.join(out, f"l_{k}.npy"), l[k])


# --------------------------------------------------------------------------- #
# Section 9.2 -- Ke-vals.py  (unit = (i, j) = (mu, rho) model index)      #
# --------------------------------------------------------------------------- #
def kevals_unit(unit):
    i, j = divmod(unit, CFG["R"])
    reps, K = CFG["reps"], 100
    mu, rho = CFG["mus"][i], CFG["rhos"][j]
    dtr = emi = umi = rmi = 0
    seed = (i * CFG["R"] + j) * reps
    for _rep in range(reps):
        cov = np.zeros((K, K))
        for p in range(K):
            for q in range(K):
                cov[p, q] = rho ** (np.abs(p - q))
        np.random.seed(seed)
        x = np.random.multivariate_normal(np.repeat(mu, K), cov)
        e = np.exp(x - 0.5)

        np.random.seed(seed)
        u = np.random.uniform(0, 1, 1)

        if np.sum(e) / K >= 1.0 / CFG["alpha"]:
            dtr += 1
        if np.sum(e) / K >= u / CFG["alpha"]:
            umi += 1

        np.random.seed(seed)
        np.random.shuffle(e)
        csum = np.cumsum(e)
        avgs = [csum[k] / (k + 1) for k in range(K)]
        if np.max(avgs) >= 1.0 / CFG["alpha"]:
            emi += 1
        if e[0] >= u / CFG["alpha"] or np.max(avgs) >= 1.0 / CFG["alpha"]:
            rmi += 1
        seed += 1
    return i, j, dtr, umi, emi, rmi


def run_kevals(outdir, workers, chunk):
    cfg = CFG
    out = os.path.join(outdir, "matrices", "Ke_vals")
    os.makedirs(out, exist_ok=True)
    res = {k: np.zeros((cfg["M"], cfg["R"])) for k in ("dtr", "umi", "emi", "rmi")}
    units = list(range(cfg["M"] * cfg["R"]))
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, j, dtr, umi, emi, rmi in pool.imap_unordered(kevals_unit, units, chunksize=1):
            res["dtr"][i, j], res["umi"][i, j] = dtr, umi
            res["emi"][i, j], res["rmi"][i, j] = emi, rmi
    for k, v in res.items():
        np.save(os.path.join(out, f"power_{k}.npy"), v)


# --------------------------------------------------------------------------- #
# Section 9.3 -- mixture_ui.py  (unit = replication index)                #
# --------------------------------------------------------------------------- #
def ui_unit(rep):
    sys.path.insert(0, HERE)
    from fmix import sample_mixture, fit_mixture, null_likelihood, split_lrt, sub_ui
    from scipy.stats import chi2

    cfg = CFG
    alpha = cfg["alpha"]
    n = 500
    pi0, v0 = cfg["pi0"], cfg["v0"]
    qchi = chi2.ppf(1 - 2.0 * alpha, 1)
    out = {k: np.zeros(cfg["n_mods"]) for k in
           ("lrt", "ui", "umi_ui", "sub_ui", "umi_sub_ui", "emi_sub_ui", "rmi_sub_ui")}
    ctr = rep * cfg["n_mods"]
    for j in range(cfg["n_mods"]):
        e = cfg["eps"][j]

        np.random.seed(ctr)
        x = sample_mixture(-e, e, v0, pi0, n)

        np.random.seed(ctr)
        u = np.random.uniform(0, 1, 1)

        (_, _, full_lik, _) = fit_mixture(x, -e, e, pi0, v0, n_starts=15)
        null_lik = null_likelihood(x, np.sum(x) / n, v0)
        lrt = np.max([0, -2 * (null_lik - full_lik)])
        out["lrt"][j] = lrt > qchi

        slrt = split_lrt(x, -e, e, pi0=pi0, v0=v0, n_starts=cfg["n_starts"])
        out["ui"][j] = slrt > 1.0 / alpha
        out["umi_ui"][j] = slrt > u / alpha

        sub_slrt = sub_ui(x, -e, e, pi0=pi0, v0=v0, n_starts=cfg["n_starts"])
        out["sub_ui"][j] = sub_slrt[-1] > 1.0 / alpha
        out["umi_sub_ui"][j] = sub_slrt[-1] > u / alpha
        out["emi_sub_ui"][j] = np.max(sub_slrt) > 1.0 / alpha
        out["rmi_sub_ui"][j] = (sub_slrt[0] > u / alpha) or (np.max(sub_slrt) > 1.0 / alpha)
        ctr += 1
    return rep, out


def run_ui(outdir, workers, chunk):
    cfg = CFG
    out = os.path.join(outdir, "matrices", "ui")
    os.makedirs(out, exist_ok=True)
    keys = ("lrt", "ui", "umi_ui", "sub_ui", "umi_sub_ui", "emi_sub_ui", "rmi_sub_ui")
    res = {k: np.zeros((cfg["n_mods"], cfg["reps"])) for k in keys}
    done = 0
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for rep, o in pool.imap_unordered(ui_unit, range(cfg["reps"]), chunksize=1):
            for k in keys:
                res[k][:, rep] = o[k]
            done += 1
            if done % 10 == 0 or done == cfg["reps"]:
                el = time.time() - t0
                print(f"[ui] {done}/{cfg['reps']} replications  {el:.0f}s elapsed"
                      f"  (eta {el / done * (cfg['reps'] - done):.0f}s)", flush=True)
    for k in keys:
        np.save(os.path.join(out, f"power_{k}.npy"), res[k])


# --------------------------------------------------------------------------- #
# Section 9.4 -- betting.py  (unit = (i, j) model index)                  #
# --------------------------------------------------------------------------- #
def betting_unit(unit):
    from confseq.betting import betting_mart
    from confseq.betting_strategies import lambda_LBOW

    cfg = CFG
    i, j = divmod(unit, cfg["n_mods"])
    n = int(cfg["ns"][i])
    a_j, b_j = cfg["a"][j], cfg["b"][j]
    alpha, reps, B = cfg["alpha"], cfg["reps"], cfg["B"]
    dtr = umi = emi = rmi = ville = 0
    ctr = (i * cfg["n_mods"] + j) * reps
    base = ctr
    for _rep in range(reps):
        np.random.seed(base + _rep)
        x = np.random.beta(a_j, b_j, n)

        mgale = betting_mart(x, 0.5, lambdas_fn_positive=lambda_LBOW)
        ville += np.max(mgale) > 1.0 / alpha

        mgales = []
        for l in range(B):
            np.random.seed(base + _rep + i * l)
            np.random.shuffle(x)
            mgales.append(betting_mart(x, 0.5, lambdas_fn_positive=lambda_LBOW)[-1])

        csum = np.cumsum(mgales)

        np.random.seed(base + _rep)
        u = np.random.uniform(0, 1, 1)[-1]
        umi += csum[-1] / B > 1.0 * u / alpha
        dtr += csum[-1] / B > 1.0 / alpha

        mgale_max = np.max([csum[b] / (b + 1) for b in range(B)])
        emi += mgale_max > 1.0 / alpha
        rmi += (csum[0] > 1.0 * u / alpha) or (mgale_max > 1.0 / alpha)
        ctr += 1
    return i, j, dtr, umi, emi, rmi, ville


def run_betting(outdir, workers, chunk):
    cfg = CFG
    out = os.path.join(outdir, "matrices", "betting")
    os.makedirs(out, exist_ok=True)
    res = {k: np.zeros((cfg["n_ns"], cfg["n_mods"])) for k in ("dtr", "umi", "emi", "rmi", "ville")}
    units = list(range(cfg["n_ns"] * cfg["n_mods"]))
    done = 0
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, j, dtr, umi, emi, rmi, ville in pool.imap_unordered(betting_unit, units, chunksize=1):
            res["dtr"][i, j], res["umi"][i, j] = dtr, umi
            res["emi"][i, j], res["rmi"][i, j] = emi, rmi
            res["ville"][i, j] = ville
            done += 1
            if done % 10 == 0 or done == len(units):
                el = time.time() - t0
                print(f"[betting] {done}/{len(units)} (n,model) cells  {el:.0f}s elapsed"
                      f"  (eta {el / done * (len(units) - done):.0f}s)", flush=True)
    for k, v in res.items():
        np.save(os.path.join(out, f"power_{k}.npy"), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=("intervals", "kevals", "betting", "ui"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n-ns", type=int, default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--n-starts", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1)
    args = ap.parse_args()

    cfg = make_cfg(args)
    _init(cfg)
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    {"intervals": run_intervals, "kevals": run_kevals,
     "betting": run_betting, "ui": run_ui}[args.experiment](args.outdir, args.workers, args.chunk)
    print(f"[par_sim] {args.experiment} done in {time.time() - t0:.1f}s "
          f"(workers={args.workers}, reps={cfg['reps']}, n_ns={cfg['n_ns']}, B={cfg['B']})",
          flush=True)


if __name__ == "__main__":
    main()
