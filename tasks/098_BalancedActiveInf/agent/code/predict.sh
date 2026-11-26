#!/bin/sh
set -eu
exec python "$BENCH_BENCH_DIR/export_policy.py" "$@"
