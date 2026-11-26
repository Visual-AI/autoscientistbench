#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$BENCH_BENCH_DIR"
export PATH="$PWD/bin:$PATH"
export ET_THREADS="$BENCH_CPUS"
export ET_TAXONOMY_DB="$BENCH_DATA_DIR/train/taxonomy/taxa.sqlite"
exec python adapter.py "$@"
