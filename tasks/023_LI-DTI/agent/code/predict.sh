#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ "$#" -eq 6 ] && [ "$1" = --model ] && [ "$3" = --input ] && [ "$5" = --output ] || { echo 'usage: predict.sh --model DIR --input CSV --output CSV' >&2; exit 2; }
exec octave-cli --no-gui --quiet bench_predict.m "$2/model.mat" "$4" "$6"
