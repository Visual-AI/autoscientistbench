#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ]
MODEL="$(realpath -m "$MODEL")"; INPUT="$(realpath -m "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${BENCH_CPUS:-16}" MKL_NUM_THREADS="${BENCH_CPUS:-16}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-16}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-16}"
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="$(mktemp -d "${TMPDIR:-/tmp}/pir-mpl.XXXXXX")"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
python -u bench_driver.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
