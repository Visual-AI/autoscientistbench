#!/usr/bin/env bash
set -euo pipefail
MODEL='' INPUT='' OUTPUT=''
while [ $# -gt 0 ]; do case "$1" in --model) MODEL=$(realpath -m "$2"); shift 2;; --input) INPUT=$(realpath -m "$2"); shift 2;; --output) OUTPUT=$(realpath -m "$2"); shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS="${BENCH_CPUS:-8}" MKL_NUM_THREADS="${BENCH_CPUS:-8}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-8}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-8}"
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$CODE"
exec python "$CODE/entry.py" predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
