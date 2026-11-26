#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR BENCH_BENCH_DIR)
# BFM-SWM has no fitted state: this only compiles the authors' mechanism, with the flags of their CMakeLists.txt.
set -euo pipefail
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }   # resolve BEFORE cd
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(abspath "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUT"
BENCH="${BENCH_BENCH_DIR:-$PWD/../bench}"
g++ -std=c++14 -O3 -w -I"$BENCH" "Influence Maximization/bench_policy.cpp" -o "$OUT/bfm_policy"
