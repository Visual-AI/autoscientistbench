#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$(realpath -m "$2")"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BENCH_SEED="${BENCH_SEED:-1}" BENCH_CPUS="${BENCH_CPUS:-8}"
python runner.py train --out "$OUT"
