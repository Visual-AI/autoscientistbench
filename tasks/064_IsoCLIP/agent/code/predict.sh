#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
abs() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }
MODEL="$(abs "$MODEL")"; INPUT="$(abs "$INPUT")"; OUTPUT="$(abs "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:$PWD/vendor"
export OMP_NUM_THREADS="${BENCH_CPUS:-16}" MKL_NUM_THREADS="${BENCH_CPUS:-16}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-16}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-16}"
exec python -u bench_predict.py --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
