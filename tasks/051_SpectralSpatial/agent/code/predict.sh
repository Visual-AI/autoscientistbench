#!/usr/bin/env bash
set -euo pipefail
model= input= output=
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2;;
    --input) input="$2"; shift 2;;
    --output) output="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n "$model" && -n "$input" && -n "$output" ]]
cd "$(dirname "${BASH_SOURCE[0]}")"
# Parallelize independent fits, with one BLAS thread per fork.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$BENCH_RUNTIME_DIR/r/bin/Rscript" reproduce/sim_driver.R "$input" "$output" "$model"
