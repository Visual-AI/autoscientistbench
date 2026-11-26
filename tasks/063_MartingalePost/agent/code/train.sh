#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == --out ]]
mkdir -p "$2"
printf 'Hybrid predictive resampling B2000 T50\n' > "$2/ready"
