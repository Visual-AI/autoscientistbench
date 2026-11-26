#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${BENCH_BENCH_DIR:-$CODE_DIR/../bench}:$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec python -u "$CODE_DIR/main.py" policy "$@"
