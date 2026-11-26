#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD"
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$(realpath ../data)}"
export BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(realpath ../bench)}"
export BENCH_CPUS="${BENCH_CPUS:-64}" BENCH_SEED="${BENCH_SEED:-1}"
export OMP_NUM_THREADS="$BENCH_CPUS" MKL_NUM_THREADS="$BENCH_CPUS" OPENBLAS_NUM_THREADS="$BENCH_CPUS"
mkdir -p "$OUT"
python "$BENCH_BENCH_DIR/resources.py" monitor --report "$OUT/resources.json" -- python -u "$PWD/main.py" train --out "$OUT"
