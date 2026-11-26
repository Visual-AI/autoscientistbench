#!/usr/bin/env bash
# bench entry point: bash predict.sh --model DIR --input PATH --output FILE   (env: BENCH_SEED BENCH_CPUS)
# For every instance in the input file: eps_0 from the bootstrap at L0 = I, bilevel learning of L (authors' optimiser),
# DRO decision w*(L*) -> CSV id,w_star,L11,L21,L22,eps.  BENCH_CPUS instances run in parallel (one thread each).
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || { echo "usage: predict.sh --model DIR --input PATH --output FILE" >&2; exit 2; }
MODEL="$(realpath -m "$MODEL")"
INPUT="$(realpath -m "$INPUT")"
OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1; the package is not installed
python bench_run.py predict --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
