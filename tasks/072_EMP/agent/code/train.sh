#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT=$(realpath -m "$OUT")
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${BENCH_CPUS:-32}" MKL_NUM_THREADS="${BENCH_CPUS:-32}" OPENBLAS_NUM_THREADS=1
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$(realpath ../data)}" BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(realpath ../bench)}"
exec python -u main.py train --out "$OUT"
