#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_DATA_DIR="$(realpath "${BENCH_DATA_DIR:-$CODE/../data}")"
export PYTHONPATH="$CODE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${BENCH_CPUS:-16}" MKL_NUM_THREADS="${BENCH_CPUS:-16}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-16}"
cd "$CODE"
exec python -u main.py train --out "$OUT"
