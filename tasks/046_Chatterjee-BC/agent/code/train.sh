#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
mkdir -p "$2"
printf '%s\n' "${BENCH_SEED:-1}" > "$2/seed.txt"
