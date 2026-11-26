#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR)
# AICoG (LB), K=9 (D=8). Epochs, scaling epochs and learning rate are the defaults in main.py.
set -euo pipefail
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }   # resolve BEFORE cd
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(abspath "$OUT")"
DATA="$(abspath "${BENCH_DATA_DIR:-../data}")"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUT"
export OMP_NUM_THREADS="${BENCH_CPUS:-4}" MKL_NUM_THREADS="${BENCH_CPUS:-4}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1
python main.py --K 9 --cuda False --seed "${BENCH_SEED:-0}" --data_dir "$DATA" --out_dir "$OUT"
