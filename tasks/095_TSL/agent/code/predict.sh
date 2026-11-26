#!/usr/bin/env bash
# bench entry point: bash predict.sh --model DIR --input PATH --output FILE
# Loads DIR/model.bin, reads the feature CSV (id + 8 raw features), writes CSV `id,pred`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo "usage: predict.sh --model DIR --input PATH --output FILE" >&2; exit 2; }
export OMP_NUM_THREADS="${BENCH_CPUS:-4}"
export RAYON_NUM_THREADS="${BENCH_CPUS:-4}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1
python main.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
