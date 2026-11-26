#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do
 case "$1" in --out) OUT="$(realpath -m "$2")"; shift 2;; *) exit 2;; esac
done
[ -n "$OUT" ] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${BENCH_CPUS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
exec python run_bench.py train --out "$OUT"
