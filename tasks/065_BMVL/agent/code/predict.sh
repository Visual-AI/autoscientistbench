#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ "$#" -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; *) exit 2;;
esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL=$(realpath "$MODEL"); INPUT=$(realpath "$INPUT"); OUTPUT=$(realpath -m "$OUTPUT")
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${BENCH_BENCH_DIR:-$PWD/../bench}:$PWD"
exec python bridge.py export --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
