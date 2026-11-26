#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT=""
while [ "$#" -gt 0 ]; do case "$1" in --out) OUT="$(realpath -m -- "$2")"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ]
export PYTHONPATH="$CODE_DIR"
export BENCH_SEED="${BENCH_SEED:-1}" BENCH_CPUS="${BENCH_CPUS:-128}"
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$(dirname "$CODE_DIR")/data}"
export BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(dirname "$CODE_DIR")/bench}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$CODE_DIR"
python "$CODE_DIR/main.py" train --out "$OUT"
