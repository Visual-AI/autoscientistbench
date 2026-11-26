#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ] || exit 2
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=true JAX_PLATFORMS=cpu
: "${BENCH_SEED:?}" "${BENCH_CPUS:?}" "${BENCH_DATA_DIR:?}"
python -u affinity_exec.py experiment.py --out "$OUT"
