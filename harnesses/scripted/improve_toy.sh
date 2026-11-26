#!/usr/bin/env bash
# A scripted "agent" for the toy-ridge task: baseline eval, improve the method, add a dependency, submit.
set -euo pipefail
cd /ws
bench status
bench eval-dev | tail -n 20
# improve: polynomial + sine features instead of raw linear features (the baseline is deliberately underfit)
python - <<'PY'
import re, pathlib
p = pathlib.Path("code/main.py"); s = p.read_text()
s = s.replace('FEATURES = "linear"', 'FEATURES = "poly_sin"')
s = s.replace("LAMBDA = 1.0", "LAMBDA = 0.05")
p.write_text(s)
PY
uv pip install --quiet tqdm==4.67.1
bench eval-dev | tail -n 20
bench submit --final -m "poly+sin features, lambda 0.05, tqdm added"
echo "scripted agent done"
