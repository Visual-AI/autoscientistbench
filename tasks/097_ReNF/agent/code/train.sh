#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
DATA="$(realpath -m "${BENCH_DATA_DIR:?}")"
mkdir -p "$OUT"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS="${BENCH_CPUS:-8}" MKL_NUM_THREADS="${BENCH_CPUS:-8}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-8}"
export CUDA_VISIBLE_DEVICES=""
python -u run_longExp.py --root_path "$DATA" --checkpoints "$OUT" \
 --data_path train.csv --model_id ETTh1_512_96 --model ReNF_alpha --data ETTh1 \
 --features M --pe 1 --revin 1 --d_layers 3 --level_dim 1 --seq_len 512 --pred_len 96 \
 --r_ema 0.996 --alpha_freq 0.0 --d_ff 2048 --gamma 0.3 --dropout 0.9 --des Exp \
 --patience 3 --train_epochs 10 --itr 1 --batch_size 128 --learning_rate 0.0006 \
 --num_workers 0 --is_training 1
