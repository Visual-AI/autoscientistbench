#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 2 && test "$1" = --out
mkdir -p "$2"
printf '%s\n' 'SNooPy is training-free; no observations or labels are read at this stage.' > "$2/model.txt"
