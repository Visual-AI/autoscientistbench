#!/usr/bin/env bash
set -euo pipefail
model=""; input=""; output=""
while (($#)); do case "$1" in
 --model) model="$2"; shift 2;; --input) input="$2"; shift 2;;
 --output) output="$2"; shift 2;; *) exit 2;; esac; done
[[ -n "$model" && -n "$input" && -n "$output" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec Rscript --vanilla "$PWD/method.R" "$model" "$input" "$output"
