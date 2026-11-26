#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "$2"
export PYTHONPATH="$PWD"
mode=full
if [[ "${BENCH_SMOKE:-0}" == 1 ]]; then mode=smoke; fi
exec python train.py "$2" "$mode"
