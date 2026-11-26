#!/usr/bin/env python3
"""Validate a policy artifact without executing candidate code."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact

def main(pred_path, input_path):
    request = json.loads(Path(input_path).read_text())
    if request.get('task') != 'icml-5412-CLMCP' or request.get('protocol') != 'private-policy-v2':
        raise ValueError('wrong public request descriptor')
    validate_artifact(pred_path, 'icml-5412-CLMCP')
    print('ok: policy artifact; decisions are produced by trusted replay')

if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception as exc:
        sys.exit(str(exc))
