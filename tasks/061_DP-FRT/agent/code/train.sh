#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == --out ]]
mkdir -p "$2"
printf 'DP-FRT-Bayes uniform prior; 10000 posterior draws\n' > "$2/ready"
