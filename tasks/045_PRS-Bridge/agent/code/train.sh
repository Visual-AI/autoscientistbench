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
export OMP_NUM_THREADS="${BENCH_CPUS:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_CPUS="${BENCH_CPUS:-16}" BENCH_SEED="${BENCH_SEED:-1}"
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$(abspath ../data)}"
# Build a private source copy so Cython never modifies the committed sources.
python - "$PWD" "$OUT" <<'PYBUILD'
import pathlib, shutil, sys
src, out = map(pathlib.Path, sys.argv[1:])
build = out / 'build_source'
build.mkdir(parents=True, exist_ok=True)
shutil.copy2(src / 'setup.py', build / 'setup.py')
shutil.copytree(src / 'bayesbridge', build / 'bayesbridge', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.pyc'))
PYBUILD
(cd "$OUT/build_source" && python setup.py build_ext --inplace) > "$OUT/build_ext.log" 2>&1
export PYTHONPATH="$OUT/build_source:$PYTHONPATH"
python bench_train.py --out "$OUT"
