#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while (( $# )); do
  case "$1" in
    --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
test -n "$MODEL" && test -n "$INPUT" && test -n "$OUTPUT"
test -f "$MODEL/model.txt"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec Rscript --vanilla "$CODE_DIR/predict.R" "$INPUT" "$OUTPUT" "$CODE_DIR"
