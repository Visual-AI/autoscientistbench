#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
THREADS=$(( ${BENCH_CPUS:-1} < 4 ? ${BENCH_CPUS:-1} : 4 ))
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONPATH="$PWD:${BENCH_BENCH_DIR:?}${PYTHONPATH:+:$PYTHONPATH}"
exec python policy.py "$@"
