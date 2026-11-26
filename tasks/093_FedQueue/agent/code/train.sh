#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR)
set -euo pipefail
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }   # resolve BEFORE cd
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(abspath "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUT"
export BENCH_CPUS="${BENCH_CPUS:-16}"
# The archived experiment uses one compute thread in each of four clients.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/src"   # harness sets PYTHONSAFEPATH=1
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$PWD/../data}" BENCH_SEED="${BENCH_SEED:-1}"
python main.py train --out "$OUT"
