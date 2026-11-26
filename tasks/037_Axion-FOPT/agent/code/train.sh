#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${BENCH_CPUS:-1}"
export OPENBLAS_NUM_THREADS="${BENCH_CPUS:-1}"
export MKL_NUM_THREADS="${BENCH_CPUS:-1}"
export NUMEXPR_NUM_THREADS="${BENCH_CPUS:-1}"
exec python "$CODE_DIR/fit.py" "$@"
