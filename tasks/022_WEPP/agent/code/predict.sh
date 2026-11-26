#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src/Freyja:$PWD"
export LD_LIBRARY_PATH="$WEPP_TOOLS_PREFIX/lib:$WEPP_NATIVE_PREFIX/lib"
exec python adapter.py "$@"
