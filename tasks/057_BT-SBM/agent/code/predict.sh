#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Dataset-level workers each use one BLAS thread, bounded by BENCH_CPUS.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec Rscript --vanilla predict.R "$@"
