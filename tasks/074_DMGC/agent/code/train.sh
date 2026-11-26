#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${BENCH_CPUS:-32}" MKL_NUM_THREADS="${BENCH_CPUS:-32}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-32}"
export BENCH_CPUS="${BENCH_CPUS:-32}" BENCH_SEED="${BENCH_SEED:-1}"
export CUDA_VISIBLE_DEVICES=""
python entry.py train --out "$OUT"
