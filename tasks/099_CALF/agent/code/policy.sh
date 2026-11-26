#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in --model) MODEL="$2";shift 2;; --input) INPUT="$2";shift 2;; --output) OUTPUT="$2";shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL="$(realpath -m "$MODEL")"; INPUT="$(realpath -m "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_DATA_DIR="$(realpath -m "${BENCH_DATA_DIR:-../data}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${BENCH_CPUS:-1}" MKL_NUM_THREADS="${BENCH_CPUS:-1}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-1}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-1}"
exec python -u bench_predict.py --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
