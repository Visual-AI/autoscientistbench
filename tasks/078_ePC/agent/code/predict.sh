#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ "$#" -gt 0 ]; do
 case "$1" in
 --model) MODEL="$(realpath -m "$2")"; shift 2;;
 --input) INPUT="$(realpath -m "$2")"; shift 2;;
 --output) OUTPUT="$(realpath -m "$2")"; shift 2;;
 *) exit 2;; esac
done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${BENCH_CPUS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
exec python run_bench.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
