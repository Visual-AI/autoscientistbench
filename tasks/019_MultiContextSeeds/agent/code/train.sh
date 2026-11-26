#!/usr/bin/env bash
# Training-free mapper: build candidate source with the copied original toolchain.
set -euo pipefail
OUT=""
while [ "$#" -gt 0 ]; do
  case "$1" in --out) OUT="$2"; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac
done
[ -n "$OUT" ] || { echo "usage: train.sh --out DIR" >&2; exit 2; }
OUT="$(realpath -m "$OUT")"
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUT"
cmake -S . -B "$OUT/build" -DCMAKE_RULE_MESSAGES=OFF --log-level=NOTICE \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" '-DCMAKE_C_FLAGS=-march=native' \
  '-DCMAKE_CXX_FLAGS=-march=native' -DZLIB_INCLUDE_DIR=/usr/include
cmake --build "$OUT/build" --parallel "${BENCH_CPUS:-12}" --target strobealign
cp "$OUT/build/strobealign" "$OUT/strobealign"
printf '%s\n' "${BENCH_SEED:-1}" > "$OUT/bench_seed.txt"
