#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
 *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ]
BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$HERE/../bench}"
python "$BENCH_BENCH_DIR/export_policy.py" --code "$HERE" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
