"""Validate an inert candidate artifact without importing its code."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact

TASK = '030_Conformal-Optimistic-Prediction'


def validate(pred_path, input_path):
    request = json.loads(Path(input_path).read_text())
    if request != {'task': TASK, 'protocol': 'private-policy-v2'}:
        raise ValueError('invalid public request')
    return validate_artifact(pred_path, TASK)


if __name__ == '__main__':
    try:
        validate(*sys.argv[1:])
        print('ok: policy artifact; intervals are recorded by trusted causal replay')
    except Exception:
        sys.exit('invalid policy artifact or request')
