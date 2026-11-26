#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${BENCH_CPUS:-8}" MKL_NUM_THREADS="${BENCH_CPUS:-8}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-8}"
python -u bench_train.py --out "$OUT"
