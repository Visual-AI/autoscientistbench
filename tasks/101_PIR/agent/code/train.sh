#!/usr/bin/env bash
set -euo pipefail
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) exit 2;; esac; done
[ -n "$OUT" ]
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_DATA_DIR="$(realpath -m "${BENCH_DATA_DIR:-../data}")"
export OMP_NUM_THREADS="${BENCH_CPUS:-16}" MKL_NUM_THREADS="${BENCH_CPUS:-16}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-16}" NUMEXPR_NUM_THREADS="${BENCH_CPUS:-16}"
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="$(mktemp -d "${TMPDIR:-/tmp}/pir-mpl.XXXXXX")"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
python -u bench_driver.py train --out "$OUT"
