#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_DATA_DIR="$(realpath -m "${BENCH_DATA_DIR:-../data}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${BENCH_CPUS:-1}" MKL_NUM_THREADS="${BENCH_CPUS:-1}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-1}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-1}"
exec python -u bench_run.py --out "$OUT"
