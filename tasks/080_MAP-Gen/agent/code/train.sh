#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ]
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${BENCH_CPUS:-16}" MKL_NUM_THREADS="${BENCH_CPUS:-16}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-16}"
python bench_run.py train --out "$OUT"
