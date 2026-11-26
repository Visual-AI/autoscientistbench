#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export BENCH_CODE_DIR="$PWD"
export BENCH_BENCH_DIR="$(realpath -m "${BENCH_BENCH_DIR:-../bench}")"
export BENCH_DATA_DIR="$(realpath -m "${BENCH_DATA_DIR:-../data}")"
exec python "$BENCH_BENCH_DIR/forecast_export.py" "$@"
