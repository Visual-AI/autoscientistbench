#!/usr/bin/env bash
set -euo pipefail
CLARI_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CLARI_CODE${PYTHONPATH:+:$PYTHONPATH}"
exec python "$CLARI_CODE/train.py" "$@"
