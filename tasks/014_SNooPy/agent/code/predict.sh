#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
exec "${BENCH_RUNTIME_DIR:?}/python/bin/python" predict.py "$@"
