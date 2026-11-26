#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
OUT="$(realpath -m "$OUT")"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_DATA_DIR="$(realpath "${BENCH_DATA_DIR:-$CODE/../data}")"
BENCH_DIR="$(realpath "${BENCH_BENCH_DIR:-$CODE/../bench}")"
export BENCH_CPUS="${BENCH_CPUS:-16}" BENCH_SEED="${BENCH_SEED:-1}"
export OMP_NUM_THREADS="$BENCH_CPUS" OPENBLAS_NUM_THREADS="$BENCH_CPUS" MKL_NUM_THREADS="$BENCH_CPUS" NUMEXPR_NUM_THREADS="$BENCH_CPUS"
export PYTHONPATH="$CODE/regime-i:$CODE"
cd "$CODE"
mkdir -p "$OUT"
python "$BENCH_DIR/run_train.py" "$CODE" "$OUT"
