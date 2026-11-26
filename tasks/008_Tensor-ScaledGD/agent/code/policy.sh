#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$BENCH_BENCH_DIR:$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec python "$CODE_DIR/tensor_policy.py" "$@"
