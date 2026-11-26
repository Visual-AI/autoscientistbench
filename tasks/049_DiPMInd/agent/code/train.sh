#!/usr/bin/env bash
set -euo pipefail
out=""
while (($#)); do case "$1" in --out) out="$2"; shift 2;; *) exit 2;; esac; done
: "${out:?--out required}"
mkdir -p "$out"
export R_MAKEVARS_USER="$PWD/Makevars.bench"
exec Rscript --vanilla "$PWD/method.R" train "$out"
