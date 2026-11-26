#!/usr/bin/env bash
set -euo pipefail
MODEL=""; INPUT=""; OUTPUT=""
while [ $# -gt 0 ]; do case "$1" in
 --model) MODEL="$2"; shift 2;; --input) INPUT="$2"; shift 2;; --output) OUTPUT="$2"; shift 2;;
 *) echo "unknown argument $1" >&2; exit 2;; esac; done
[ -n "$MODEL" ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ] || exit 2
MODEL="$(realpath -m "$MODEL")"; INPUT="$(realpath -m "$INPUT")"; OUTPUT="$(realpath -m "$OUTPUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD"
export BENCH_CODE_DIR="$PWD"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${BENCH_CPUS:-8}" MKL_NUM_THREADS="${BENCH_CPUS:-8}" OPENBLAS_NUM_THREADS="${BENCH_CPUS:-8}"
python -u "${BENCH_BENCH_DIR:?}/predict_runner.py" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
