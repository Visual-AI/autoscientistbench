#!/usr/bin/env bash
set -euo pipefail
out=""
while (($#)); do case "$1" in --out) out="$2"; shift 2;; *) exit 2;; esac; done
: "${out:?--out required}"
mkdir -p "$out"
printf '%s\n' 'ALSE Section 5: per-dataset nuisance fitting in predict; gam,ranger; 5 folds; 1000 trees' > "$out/model.txt"
