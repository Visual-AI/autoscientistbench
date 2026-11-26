#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do
  case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;; esac
done
[ -n "$OUT" ] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
printf 'training_free\nseed=%s\ncpus=%s\n' "${BENCH_SEED:-1}" "${BENCH_CPUS:-1}" > "$OUT/NO_TRAINING"
