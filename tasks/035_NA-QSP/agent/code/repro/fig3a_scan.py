#!/usr/bin/env python
"""Run the main experiment of the paper: the alpha scan behind Fig. 3(a) / Table II.

The main experiment is the hybrid oscillator-qubit (CV-DV) simulation of the
Gaussian-controlled rotation (GCR) versus the classical composite pulse BB1:
an equispaced scan over the oscillator mean position |alpha| = 2.00 ... 9.95
(160 points, chi = pi/(4|alpha|)), theta_t = pi/2, Delta = 1, truncated Fock
space Ncav = 400, in which for every alpha we record

  * failure probability      Pe      = 1 - |<g|U_gg|g>|^2          (Eq. 20)
  * hybrid infidelity        1 - FH  = infidelity of the full qubit+oscillator
                                       state with the ideal target state (Eq. 21)
  * post-selected infidelity 1 - FH|g (infidelity after heralding on |g>)

for both pulse sequences (paper Eqs. 30 and 29).  Table II of the paper quotes
the alpha = 2 point (the first point of Fig. 3).

Implementation note: instead of re-typing the physics we execute the authors'
own notebook cells verbatim (NA-QSP_sims.ipynb, code cells 0,1,3,4,5 = the
"Fig. 2a and Fig. 14 [Performance of GCR]" block, which is the code that
generates the data of Fig. 3(a) and Table II).  The only edits applied to the
extracted source are (i) redirecting the output path "Paper_Data/" to the
output directory so that the authors' shipped data files are never overwritten
and the headline metric is computed from freshly generated data, and (ii) the
--ncav / --npts overrides for convergence checks and pilot runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]           # .../Phy-11_NA-QSP_sims
REPO = ROOT / "code"
NOTEBOOK = REPO / "NA-QSP_sims.ipynb"

# code cells of NA-QSP_sims.ipynb that produce the Fig. 3(a) / Table II data
MAIN_CELLS = [0, 1, 3, 4, 5]
CELL_LABELS = {
    0: "imports",
    1: "qubit operators",
    3: "Ncav=400 operator units",
    4: "GCR: 160-point alpha scan",
    5: "BB1: 160-point alpha scan",
}


def notebook_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text())["cells"]


def cell_source(cells: list[dict], idx: int) -> str:
    rec = cells[idx]
    if rec["cell_type"] != "code":
        raise SystemExit(f"notebook cell {idx} is not a code cell")
    return "".join(rec["source"])


def patch(src: str, outdir: Path, ncav: int, npts: int | None) -> str:
    """Redirect the notebook's output path and apply the run-size overrides."""
    src = src.replace("Paper_Data/", str(outdir) + "/")
    if npts is not None:
        # cells 4 and 5 scan  for alpha1 in range(40,200):  alpha = alpha1/20
        src, n = re.subn(r"range\(40,\s*200\)", f"range(40,{40 + npts})", src)
        if n == 0:
            raise SystemExit("could not shorten the scan: range(40,200) not found")
    return re.sub(r"^Ncav\s*=\s*400", f"Ncav = {ncav}", src, flags=re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncav", type=int, default=400,
                    help="Fock-space truncation (paper: 400)")
    ap.add_argument("--npts", type=int, default=160,
                    help="number of alpha points (paper: 160, alpha = 2.00..9.95)")
    ap.add_argument("--out", default="results",
                    help="output directory for the .npz data (default: results/)")
    args = ap.parse_args()

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    npts = None if args.npts == 160 else args.npts

    cells = notebook_cells()
    hashes = {str(i): hashlib.sha256(cell_source(cells, i).encode()).hexdigest()[:16]
              for i in MAIN_CELLS}

    print("=" * 78)
    print("NA-QSP Fig. 3(a) / Table II alpha scan  --  fresh simulation")
    print("=" * 78)
    print(f"repo                : {REPO}")
    print(f"notebook            : {NOTEBOOK.name} cells {MAIN_CELLS} (sha256_16: {hashes})")
    print(f"Ncav                : {args.ncav}   (paper: 400)")
    print(f"alpha points        : {args.npts}   alpha = 2.00 ... {(40 + args.npts - 1) / 20:.2f}")
    print("sequence            : theta_t = pi/2, Delta = 1, chi = pi/(4|alpha|)")
    print(f"output directory    : {outdir}")
    print("-" * 78, flush=True)

    ns: dict = {"__name__": "__main__"}
    timings: dict[str, float] = {}
    t_all = time.time()
    for idx in MAIN_CELLS:
        src = patch(cell_source(cells, idx), outdir, args.ncav,
                    npts if idx in (4, 5) else None)
        print(f"[cell {idx}] {CELL_LABELS[idx]} ...", flush=True)
        t0 = time.time()
        exec(compile(src, f"NA-QSP_sims.ipynb#cell{idx}", "exec"), ns)
        timings[str(idx)] = time.time() - t0
        print(f"[cell {idx}] done in {timings[str(idx)]:.1f} s", flush=True)
    wall = time.time() - t_all

    alph, fail_prob, fail_b = ns["alph"], ns["fail_prob"], ns["fail_b"]
    print("-" * 78)
    print(f"alpha grid          : {alph[0]:.2f} ... {alph[-1]:.2f}  (n={len(alph)})")
    print(f"alpha=2 (Table II)  : Pe(GCR)={fail_prob[0]:.4e}   Pe(BB1)={fail_b[0]:.4e}")
    print(f"total wall time     : {wall:.1f} s")

    import qutip
    meta = {
        "stage": "train",
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "notebook_cells": MAIN_CELLS,
        "notebook_cell_sha256_16": hashes,
        "ncav": args.ncav,
        "npts": args.npts,
        "alpha_range": [float(alph[0]), float(alph[-1])],
        "theta": float(ns["theta"]),
        "Delta": float(ns["Delta"]),
        "wall_s": wall,
        "cell_wall_s": timings,
        "outputs": sorted(p.name for p in outdir.glob("*.npz")),
        "python": sys.version.split()[0],
        "qutip": qutip.__version__,
        "platform": platform.platform(),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {outdir / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
