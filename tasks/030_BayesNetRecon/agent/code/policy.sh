#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}:${BENCH_BENCH_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${BENCH_CPUS:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec python -u policy.py "$@"
