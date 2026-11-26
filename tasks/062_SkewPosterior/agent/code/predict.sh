#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export R_MAKEVARS_USER="$PWD/Makevars"
export STAN_NUM_THREADS=1 RCPP_PARALLEL_NUM_THREADS=1
Rscript --vanilla predict.R "$@"
