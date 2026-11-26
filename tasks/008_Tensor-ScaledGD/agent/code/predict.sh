#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL=""; OUTPUT=""
while [ "$#" -gt 0 ]; do case "$1" in --model) MODEL="$2";shift 2;; --input) shift 2;; --output) OUTPUT="$2";shift 2;; *) exit 2;; esac; done
export PYTHONPATH="$BENCH_BENCH_DIR:$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec python "$BENCH_BENCH_DIR/tensor_export.py" --code "$CODE_DIR" --model "$MODEL" --output "$OUTPUT"
