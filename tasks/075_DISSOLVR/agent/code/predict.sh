#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
 *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo 'usage: predict.sh --model DIR --input PATH --output FILE' >&2; exit 2; }
MODEL="$(realpath "$MODEL")"; INPUT="$(realpath "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_CPUS="${BENCH_CPUS:-16}" BENCH_SEED="${BENCH_SEED:-1}"
export OMP_NUM_THREADS="$BENCH_CPUS" OPENBLAS_NUM_THREADS="$BENCH_CPUS" MKL_NUM_THREADS="$BENCH_CPUS" NUMEXPR_NUM_THREADS="$BENCH_CPUS"
export PYTHONPATH="$CODE/regime-i:$CODE"
cd "$CODE"
python "$CODE/regime-i/eval_test.py" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
