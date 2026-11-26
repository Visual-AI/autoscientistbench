#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${BENCH_CPUS:-1}" MKL_NUM_THREADS="${BENCH_CPUS:-1}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-1}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
python bench_run.py train --out "$OUT"
