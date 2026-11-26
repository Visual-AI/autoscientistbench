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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # harness sets PYTHONSAFEPATH=1
python bench_run.py train --out "$OUT"
