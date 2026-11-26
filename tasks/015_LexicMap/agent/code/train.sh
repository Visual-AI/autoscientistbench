#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(dirname "${BASH_SOURCE[0]}")"
exec /usr/bin/python3 "$(dirname "${BASH_SOURCE[0]}")/adapter.py" train "$@"
