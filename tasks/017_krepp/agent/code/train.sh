#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ "$#" -eq 2 && "$1" == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
out="$2"
mkdir -p "$out"
export PYTHONPATH="$PWD"
# No fitting: compile the candidate source; never use a copied placement executable.
make -j"${BENCH_CPUS:-1}" COMPILER="$PWD/compiler.sh"
cp krepp "$out/krepp"
python3 - "$out/build.json" <<'INNER'
import json,os,sys
json.dump({'seed':int(os.environ['BENCH_SEED']),'cpus':int(os.environ['BENCH_CPUS']),'method':'krepp v0.11.1','training':'none; candidate source compilation'},open(sys.argv[1],'w'))
INNER
