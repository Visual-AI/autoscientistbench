#!/usr/bin/env bash
# bench entry point: bash predict.sh --model DIR --input PATH --output FILE
set -euo pipefail
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }   # resolve BEFORE cd
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo "usage: predict.sh --model DIR --input PATH --output FILE" >&2; exit 2; }
MODEL="$(abspath "$MODEL")"; INPUT="$(abspath "$INPUT")"; OUTPUT="$(abspath "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export BENCH_CPUS="${BENCH_CPUS:-16}"
# The archived experiment uses one compute thread in each of four clients.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/src"   # harness sets PYTHONSAFEPATH=1
python main.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
