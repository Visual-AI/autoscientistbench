#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RSCRIPT:?snapshot runtime must set RSCRIPT}"
exec "$RSCRIPT" --vanilla "$code_dir/predict.R" "$@"
