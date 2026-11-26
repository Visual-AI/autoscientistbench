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
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo 'usage: predict.sh --model DIR --input PATH --output FILE' >&2; exit 2; }
abs() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }
MODEL="$(abs "$MODEL")"; INPUT="$(abs "$INPUT")"; OUTPUT="$(abs "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export BENCH_CPUS="${BENCH_CPUS:-1}" BENCH_SEED="${BENCH_SEED:-1}"
# Parallelism is across cores; prevent each worker from spawning extra BLAS threads.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export MPLBACKEND=Agg
python bench_predict.py --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
