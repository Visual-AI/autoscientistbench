#!/usr/bin/env python3
import sys
from pathlib import Path
from common import load_json, validate_prediction

try:
    _, count = validate_prediction(sys.argv[1], load_json(Path(sys.argv[2])/'manifest.json')['contigs'])
    print('ok:',count,'VCF records')
except Exception as exc:
    sys.exit(str(exc))
