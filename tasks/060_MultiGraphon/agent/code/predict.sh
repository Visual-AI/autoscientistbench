#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL="$(realpath -m "$MODEL")"
INPUT="$(realpath -m "$INPUT")"
OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
# BENCH_CPUS independent workers; a single numeric thread in each worker.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
python bench_run.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
