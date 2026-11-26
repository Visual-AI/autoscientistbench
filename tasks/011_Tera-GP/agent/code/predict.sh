#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL="$(realpath -m "$MODEL")"; INPUT="$(realpath "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD"
export BENCH_BENCH_DIR="${BENCH_BENCH_DIR:-$(realpath ../bench)}"
export BENCH_CPUS="${BENCH_CPUS:-64}" BENCH_SEED="${BENCH_SEED:-1}"
export OMP_NUM_THREADS="$BENCH_CPUS" MKL_NUM_THREADS="$BENCH_CPUS" OPENBLAS_NUM_THREADS="$BENCH_CPUS"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python "$BENCH_BENCH_DIR/resources.py" monitor --report "$TMP/resources.json" -- python -u "$PWD/main.py" predict --model "$MODEL" --input "$INPUT" --output "$TMP/raw.csv"
python "$BENCH_BENCH_DIR/resources.py" format --raw "$TMP/raw.csv" --output "$OUTPUT" --training "$MODEL/resources.json" --prediction "$TMP/resources.json"
