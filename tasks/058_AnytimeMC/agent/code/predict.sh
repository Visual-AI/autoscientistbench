#!/bin/sh
set -eu
exec python3 "$BENCH_BENCH_DIR/export_policy.py" "$@"
