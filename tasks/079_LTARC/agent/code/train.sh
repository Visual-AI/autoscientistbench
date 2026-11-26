#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${BENCH_BENCH_DIR:-$CODE_DIR/../bench}:$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python "$CODE_DIR/adapter.py" train "$@"
