#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$(realpath ../data)}" BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(realpath ../bench)}"
export PYTHONPATH="$PWD/src:$PWD"
THREADS="${BENCH_CPUS:-8}"; if [ "$THREADS" -gt 2 ]; then THREADS=2; fi
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
python -u main.py train --out "$OUT"
