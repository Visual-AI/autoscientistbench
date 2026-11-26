#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "$2"
export PYTHONPATH="$PWD"
exec python make_local_cwl.py "$2/cwl"
