#!/usr/bin/env python3
"""Check inert policy artifacts; scientific certificates are checked during replay."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact


def validate(pred_path, input_path):
    request = json.loads(Path(input_path).read_text())
    if request.get('task') != 'icml-3290-LTARC' or request.get('protocol') != 'private-policy-v2':
        raise ValueError('invalid public request')
    validate_artifact(pred_path, 'icml-3290-LTARC')


if __name__ == '__main__':
    try:
        validate(*sys.argv[1:])
        print('ok: inert policy archive; private replay verifies FFT certificates')
    except Exception:
        sys.exit('invalid policy archive or public request')
