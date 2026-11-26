#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$#" == 2 && "$1" == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
mkdir -p "$2"
python3 - "$2" <<'PYTRAIN'
import json, os, pathlib, sys
seed=int(os.environ['BENCH_SEED'])
pathlib.Path(sys.argv[1],'protocol.json').write_text(json.dumps({'seed':seed,'method':'TS_IG_network_DMFM_BNB','train':'no fitting; posterior inference is performed on each supplied network in predict'}))
print('Training-free per-network Bayesian inference: model protocol saved.')
PYTRAIN
