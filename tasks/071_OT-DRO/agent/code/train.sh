#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR)
# The paper's method has no training stage: L is learned per instance at prediction time (see predict.sh).
# This step only records the method configuration (Table-1 hyper-parameters) that predict.sh reads.
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUT"
export OMP_NUM_THREADS="${BENCH_CPUS:-8}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1; the package is not installed
python bench_run.py train --out "$OUT"
