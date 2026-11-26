#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "$R_HOME/bin/Rscript" --vanilla predict.R "$@"
