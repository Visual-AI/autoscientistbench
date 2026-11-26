#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR)
set -euo pipefail
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }   # resolve BEFORE cd
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(abspath "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUT"
THREADS=$(( ${BENCH_CPUS:-1} < 4 ? ${BENCH_CPUS:-1} : 4 ))
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1
python -u bench_main.py train --out "$OUT"
