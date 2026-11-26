#!/usr/bin/env bash
set -euo pipefail
OUT=""
while (( $# )); do
  case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;; esac
done
test -n "$OUT"
mkdir -p "$OUT"
printf '%s\n' 'Geodesic DiD is an inference-only estimator; predict.sh fits each supplied observed dataset.' > "$OUT/model.txt"
