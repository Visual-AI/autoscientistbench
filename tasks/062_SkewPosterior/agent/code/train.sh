#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ $# -eq 2 && "$1" == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
mkdir -p "$2"
export R_MAKEVARS_USER="$PWD/Makevars"
export STAN_NUM_THREADS=1 RCPP_PARALLEL_NUM_THREADS=1
Rscript --vanilla prepare.R "$2"
