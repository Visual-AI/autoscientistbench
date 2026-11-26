#!/usr/bin/env bash
set -euo pipefail
COP_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${BENCH_BENCH_DIR:-$COP_CODE/../bench}:$COP_CODE${PYTHONPATH:+:$PYTHONPATH}"
exec python "$COP_CODE/adapter.py" policy "$@"
