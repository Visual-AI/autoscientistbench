#!/usr/bin/env bash
# bench entry point: bash train.sh --out DIR   (env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR)
# Baseline = the paper's unified setting for ETTh1 (scripts/xPatch_unified.sh): lookback 96, one model per horizon
# T in {96,192,336,720}, batch 2048, AdamW lr 5e-4 with the sigmoid schedule, EMA decomposition alpha = 0.3,
# arctangent-weighted MAE loss, <= 100 epochs, early stopping on the validation loss with patience 10.
# Leaves DIR/h<T>/<setting>/{checkpoint.pth,meta.json} (best-on-val weights + the train-split scaler) for predict.sh.
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(realpath -m "$OUT")"
mkdir -p "$OUT"
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${BENCH_CPUS:-8}" MKL_NUM_THREADS="${BENCH_CPUS:-8}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1
DATA="$(realpath -m "${BENCH_DATA_DIR:-../data}")"
for T in 96 192 336 720; do
  echo "=== horizon $T (seed ${BENCH_SEED:-2021}, threads ${BENCH_CPUS:-8}) ==="
  python -u run.py \
    --is_training 1 \
    --root_path "$DATA" \
    --data_path ETTh1.csv \
    --model_id ETTh1_${T}_ema \
    --model xPatch \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len $T \
    --enc_in 7 \
    --des Exp \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj sigmoid \
    --ma_type ema \
    --alpha 0.3 \
    --beta 0.3 \
    --train_epochs 100 \
    --patience 10 \
    --num_workers 0 \
    --checkpoints "$OUT/h$T"
done
