#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:$PWD/src"
export BENCH_SEED="${BENCH_SEED:-1}" BENCH_CPUS="${BENCH_CPUS:-128}"
export BENCH_DATA_DIR="$(realpath -m "${BENCH_DATA_DIR:-../data}")"
exec python -u launch.py "$OUT"
