#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ]
MODEL="$(realpath -m "$MODEL")"; INPUT="$(realpath -m "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD"
export MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=1073741824
export OMP_NUM_THREADS="${BENCH_CPUS:-1}" MKL_NUM_THREADS="${BENCH_CPUS:-1}"
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -u bench_run.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
