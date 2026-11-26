#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ "$#" -eq 2 ] && [ "$1" = --out ] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
mkdir -p "$2"
fixture="$BENCH_DATA_DIR/train/full.mat"
if [ "${BENCH_SMOKE:-0}" = 1 ]; then fixture="$BENCH_DATA_DIR/train/smoke.mat"; fi
exec octave-cli --no-gui --quiet bench_train.m "$fixture" "$2/model.mat"
