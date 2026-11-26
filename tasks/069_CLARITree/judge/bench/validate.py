#!/usr/bin/env python3
"""Validate only the inert archive; the scorer validates all actual fitted trees."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact

def main(pred_path, input_path):
    if json.loads(Path(input_path).read_text()) != {'task': 'icml-2217-CLARITree', 'protocol': 'private-policy-v2'}:
        raise ValueError('wrong public request')
    validate_artifact(pred_path, 'icml-2217-CLARITree')
    print('ok: inert policy artifact; fitted-tree checks run in the trusted scorer')

if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception:
        sys.exit('invalid policy artifact or request')
