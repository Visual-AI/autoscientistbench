#!/usr/bin/env bash
set -euo pipefail
model=""; input=""; output=""
while (($#)); do case "$1" in
  --model) model="$2"; shift 2;; --input) input="$2"; shift 2;;
  --output) output="$2"; shift 2;; *) exit 2;; esac; done
: "${model:?--model required}" "${input:?--input required}" "${output:?--output required}"
exec Rscript --vanilla "$BENCH_BENCH_DIR/runner.R" "$PWD" "$model" "$input" "$output"
