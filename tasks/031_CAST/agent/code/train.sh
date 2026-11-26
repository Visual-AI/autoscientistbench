#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
export BENCH_DATA_DIR="$(realpath -m "${BENCH_DATA_DIR:-$(dirname "$0")/../data}")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export BENCH_CPUS="${BENCH_CPUS:-16}" BENCH_SEED="${BENCH_SEED:-1}"
# Independent workers share the CPU budget; each numerical library uses one thread.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_THREAD_LIMIT=1
exec python -u bench_main.py train --out "$OUT"
