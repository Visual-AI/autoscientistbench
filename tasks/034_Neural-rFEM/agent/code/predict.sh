#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
 *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL="$(realpath -m "$MODEL")"; INPUT="$(realpath -m "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${BENCH_CPUS:-32}" MKL_NUM_THREADS="${BENCH_CPUS:-32}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-32}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python "$CODE/predict_mesh.py" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
