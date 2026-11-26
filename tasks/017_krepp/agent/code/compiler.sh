#!/usr/bin/env bash
set -euo pipefail
r="${BENCH_RUNTIME_DIR:?}/toolchain"
exec "$r/usr/bin/g++" --sysroot="$r" -B"$r/usr/bin/" -B"$r/usr/lib/gcc/x86_64-linux-gnu/12/" "$@"
