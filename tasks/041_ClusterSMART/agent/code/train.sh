#!/usr/bin/env bash
set -euo pipefail
out=""
while (( $# )); do
 case "$1" in --out) out="$2"; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac
done
[ -n "$out" ] || { echo '--out required' >&2; exit 2; }
mkdir -p "$out"
printf '%s\n' '{"method":"authors longitudinal SMART estimator","training_free":true}' > "$out/model.json"
