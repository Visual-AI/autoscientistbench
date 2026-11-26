#!/usr/bin/env bash
set -euo pipefail
out=""
while (($#)); do case "$1" in --out) out="$2"; shift 2;; *) exit 2;; esac; done
[[ -n "$out" ]] || exit 2
mkdir -p "$out"
printf '%s\n' 'No fitting: original anytime-valid BC method, h=10, b=0.9.' > "$out/method.txt"
