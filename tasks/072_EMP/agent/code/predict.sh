#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ "$#" -gt 0 ]; do case "$1" in --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL=$(realpath "$MODEL"); INPUT=$(realpath "$INPUT"); OUTPUT=$(realpath -m "$OUTPUT")
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${BENCH_CPUS:-32}" MKL_NUM_THREADS="${BENCH_CPUS:-32}" OPENBLAS_NUM_THREADS=1
export BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(realpath ../bench)}"
exec python -u main.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
