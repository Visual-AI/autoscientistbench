#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/reproduced:$BENCH_BENCH_DIR"
exec python bench_entry.py predict "$@"
