#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$#" == 2 && "$1" == --out ]] || { echo 'usage: train.sh --out DIR' >&2; exit 2; }
mkdir -p "$2"
python3 - "$2" <<'PYTHON'
import json,os,sys
from pathlib import Path
Path(sys.argv[1], 'protocol.json').write_text(json.dumps({'method':'hFDR fixed-X Gaussian Lasso','seed':int(os.environ.get('BENCH_SEED','1'))}))
PYTHON
