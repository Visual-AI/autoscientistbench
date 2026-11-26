#!/usr/bin/env bash
set -euo pipefail
out=""
while (($#)); do case "$1" in --out) out="$2"; shift 2;; *) exit 2;; esac; done
: "${out:?--out required}"
mkdir -p "$out"
printf '%s\n' 'reservoir-UPAVE k=500 L=500 m=2500 R=50' > "$out/model.txt"
