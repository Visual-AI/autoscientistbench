#!/usr/bin/env python3
"""Thin wrapper around the authors' official implementation (Paper_Code/Cir_Sym/debias_prog.py).

Two things happen here, both documented in REPRODUCTION.md:

1. `debias_prog.py` starts with `import ray` (used only by its unused `DualComp_Ray`
   helper).  Ray is not needed for the circulant-symmetric experiment, so a stub
   module is registered in `sys.modules` before importing the authors' file.  The
   authors' file itself is NOT modified.

2. `DebiasProg_SCSfirst` is a verbatim copy of the authors' `DebiasProg` with the
   solver fallback list reordered to try SCS before ECOS.  Both are entries of the
   authors' own fallback chain and solve the identical convex program; on this node
   ECOS needs ~40-55 s per solve (dense KKT factorisation with 1000 dense
   constraints) while SCS needs ~3.5 s, which is the difference between ~5 and ~45
   Monte-Carlo replications inside the 1-hour training budget.  All other functions
   (ScaledLasso, DualCD, DualObj, ...) are the authors' code, unmodified.
"""
import os
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIRS_SYM_DIR = os.path.join(REPO, "Paper_Code", "Cir_Sym")

# ---- (1) ray stub -----------------------------------------------------------
_ray = types.ModuleType("ray")


def _remote(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f


_ray.remote = _remote
sys.modules.setdefault("ray", _ray)

if CIRS_SYM_DIR not in sys.path:
    sys.path.insert(0, CIRS_SYM_DIR)

import cvxpy as cp  # noqa: E402
from debias_prog import ScaledLasso, DualCD, DualObj  # noqa: E402  (authors' code)

# ---- (2) authors' DebiasProg with SCS tried first ---------------------------
SOLVER_ORDER = "SCS,ECOS,CVXOPT"  # authors' order is "ECOS,SCS,CVXOPT"


def DebiasProg_SCSfirst(X, x, Pi, gamma_n=0.1):
    """Same program/tolerances as the authors' DebiasProg; solver order SCS first."""
    n = X.shape[0]
    w = cp.Variable(n)
    debias_obj = cp.Minimize(cp.quad_form(w, Pi))
    constraints = [x - (1 / np.sqrt(n)) * (w @ Pi @ X) <= gamma_n,
                   x - (1 / np.sqrt(n)) * (w @ Pi @ X) >= -gamma_n]
    debias_prog = cp.Problem(debias_obj, constraints)

    solved = False
    for solver, kwargs in [(cp.SCS, {"max_iters": 20000, "eps": 1e-5}),
                           (cp.ECOS, {}),
                           (cp.CVXOPT, {"max_iters": 30000})]:
        try:
            debias_prog.solve(solver=solver, **kwargs)
            if debias_prog.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                solved = True
                break
        except Exception as e:  # pragma: no cover - mirrors the authors' control flow
            print(f"Solver {solver} failed: {repr(e)}")

    if not solved:
        return np.full(X.shape[0], np.nan)

    return np.asarray(w.value).reshape(-1)


__all__ = ["ScaledLasso", "DualCD", "DualObj", "DebiasProg_SCSfirst", "SOLVER_ORDER"]
