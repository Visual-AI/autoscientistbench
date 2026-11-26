#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS="${BENCH_CPUS:-32}" MKL_NUM_THREADS="${BENCH_CPUS:-32}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-32}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python "$CODE/train_entry.py" --out "$OUT"
