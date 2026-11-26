#!/usr/bin/env bash
set -euo pipefail
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model|--input|--output) ARGS+=("$1" "$(realpath -m "$2")"); shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
# Parallelize instances, keeping BLAS and OpenMP single-threaded in each worker.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export MPLBACKEND=Agg
python bench_run.py predict "${ARGS[@]}"
