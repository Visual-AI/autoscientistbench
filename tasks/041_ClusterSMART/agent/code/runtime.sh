#!/usr/bin/env bash
set -euo pipefail
runtime_root="${BENCH_RUNTIME_DIR:?BENCH_RUNTIME_DIR must point to the frozen task runtime}"
export R_HOME="$runtime_root/r/lib/R"
export R_LIBS_USER="$runtime_root/r/lib/R/library"
export LD_LIBRARY_PATH="$runtime_root/r/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$runtime_root/r/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
