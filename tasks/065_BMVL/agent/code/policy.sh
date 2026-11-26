#!/usr/bin/env bash
set -euo pipefail
exec python /ws/code/bridge.py policy "$@"
