#!/usr/bin/env bash
# One instance in one isolated worker: bash policy.sh --model DIR --input INSTANCE --output FILE
# (env: BENCH_RPC_SOCKET BENCH_DATA_DIR BENCH_SEED BENCH_CPUS BENCH_BENCH_DIR)
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
exec "$MODEL/bfm_policy" --input "$INPUT" --output "$OUTPUT"
