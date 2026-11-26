#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT=$(realpath -m "$OUT")
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:${BENCH_BENCH_DIR:-$PWD/../bench}"
exec python bridge.py train --out "$OUT"
