#!/usr/bin/env bash
set -euo pipefail
out=''
while (($#)); do case "$1" in --out) out="$2"; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
[[ -n "$out" ]] || exit 2
mkdir -p "$out"
printf '%s\n' "${BENCH_SEED:-1}" > "$out/seed.txt"
# This estimator fits each independent observed learning set during prediction.
