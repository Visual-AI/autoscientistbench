#!/usr/bin/env bash
# Per-trace entry point, started by the trusted scorer in an isolated worker.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec python "$HERE/policy.py" "$@"
