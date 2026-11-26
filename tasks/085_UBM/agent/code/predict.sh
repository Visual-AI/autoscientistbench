#!/usr/bin/env bash
set -euo pipefail
MODEL=''; INPUT=''; OUTPUT=''
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$(realpath -m "$2")"; shift 2;;
 --input) INPUT="$(realpath -m "$2")"; shift 2;;
 --output) OUTPUT="$(realpath -m "$2")"; shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
export PYTHONPATH="$PWD"
export BENCH_SEED="${BENCH_SEED:-1}" BENCH_CPUS="${BENCH_CPUS:-8}"
export BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(realpath ../bench)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec python -u bench_run.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
