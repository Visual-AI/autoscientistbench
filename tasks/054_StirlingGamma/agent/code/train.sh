#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $# == 2 && $1 == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
out="$2"
mkdir -p "$out/Rlib" "$out/source"
# Build the editable candidate package, never the installed snapshot package.
for item in DESCRIPTION NAMESPACE LICENSE R src man; do cp -R "$item" "$out/source/"; done
find "$out/source/src" -type f \( -name '*.o' -o -name '*.so' \) -delete
R CMD INSTALL --preclean --no-multiarch --library="$out/Rlib" "$out/source"
printf '%s\n' "candidate ConjugateDP source rebuilt; BENCH_SEED=${BENCH_SEED:-1}" > "$out/build.txt"
