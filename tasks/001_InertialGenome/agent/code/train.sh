#!/usr/bin/env bash
set -euo pipefail
OUT=''
while [ $# -gt 0 ]; do case "$1" in --out) OUT=$(realpath -m "$2"); shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS="${BENCH_CPUS:-8}" MKL_NUM_THREADS="${BENCH_CPUS:-8}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-8}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-8}"
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$CODE"
exec python "$CODE/entry.py" train --out "$OUT"
