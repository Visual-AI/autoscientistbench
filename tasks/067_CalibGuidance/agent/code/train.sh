#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do
  case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac
done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
mkdir -p "$OUT"
printf '%s\n' "${BENCH_SEED:-1}" > "$OUT/NO_TRAINING"
