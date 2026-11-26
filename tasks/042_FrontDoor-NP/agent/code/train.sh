#!/usr/bin/env bash
set -euo pipefail
out=""
while (($#)); do
  case "$1" in --out) out="$2"; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;; esac
done
test -n "$out"
mkdir -p "$out"
printf '%s\n' '{"method":"front-door TMLE","training":"per-input-dataset estimation"}' > "$out/model.json"
