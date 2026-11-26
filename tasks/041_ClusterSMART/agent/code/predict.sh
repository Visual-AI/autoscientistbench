#!/usr/bin/env bash
set -euo pipefail
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$code_root/runtime.sh"
exec Rscript --vanilla "$code_root/predict.R" "$@"
