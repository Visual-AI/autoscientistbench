#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR)
# Trains TSL (R<=2) on $BENCH_DATA_DIR/train/train.csv with the authors' released best hyperparameters
# (configs/california_housing_tsl_r2.json) and writes model.bin + model_card.json + train_meta.json into DIR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
mkdir -p "$OUT"
export OMP_NUM_THREADS="${BENCH_CPUS:-4}"
export RAYON_NUM_THREADS="${BENCH_CPUS:-4}"          # tensorsl's Rust core (bagged fits) parallelises with rayon
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1
python main.py train --data "${BENCH_DATA_DIR:-../data}" --seed "${BENCH_SEED:-0}" --out "$OUT"
