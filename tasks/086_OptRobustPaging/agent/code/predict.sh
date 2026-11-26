#!/usr/bin/env bash
# bench entry point: bash predict.sh --model DIR --input PATH --output FILE
# Seals code/ and the model into a policy artifact. No trace is read here: the scorer replays
# the artifact, starting code/policy.sh once per trace in an isolated worker.
set -euo pipefail
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }   # resolve BEFORE cd
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo "usage: predict.sh --model DIR --input PATH --output FILE" >&2; exit 2; }
MODEL="$(abspath "$MODEL")"; INPUT="$(abspath "$INPUT")"; OUTPUT="$(abspath "$OUTPUT")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${BENCH_BENCH_DIR:-$HERE/../bench}/export_policy.py" --code "$HERE" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
