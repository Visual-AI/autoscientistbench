#!/usr/bin/env bash
# bench entry point: bash predict.sh --model DIR --input REQUEST --output ARTIFACT
# Seals code/ and DIR into a policy artifact; instances run later, one isolated policy.sh per instance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo "usage: predict.sh --model DIR --input REQUEST --output ARTIFACT" >&2; exit 2; }
BENCH="${BENCH_BENCH_DIR:-$HERE/../bench}"
exec python "$BENCH/export_policy.py" --code "$HERE" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
