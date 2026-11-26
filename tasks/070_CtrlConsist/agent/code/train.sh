#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu WANDB_MODE=disabled
export OMP_NUM_THREADS="${BENCH_CPUS:-16}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-16}"
export TQDM_MININTERVAL=30
python bench_run.py train --out "$OUT"
