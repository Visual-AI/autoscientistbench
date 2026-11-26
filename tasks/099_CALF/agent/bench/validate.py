"""Validate the submitted policy archive without executing candidate code."""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from private_eval import validate_artifact


def validate(prediction, input_path):
    config = json.loads((HERE / 'forecast_config.json').read_text())
    descriptor = json.loads(Path(input_path).read_text())
    if descriptor.get('protocol') != 'private-forecast-v2' or descriptor.get('task_id') != config['task_id']:
        raise ValueError('wrong task descriptor')
    if set(descriptor) != {'protocol', 'task_id', 'windows', 'horizons', 'windows_per_horizon', 'channels'}:
        raise ValueError('descriptor must not carry observations or labels')
    if descriptor['windows'] < 1 or descriptor['channels'] != 7:
        raise ValueError('invalid descriptor dimensions')
    return validate_artifact(prediction, config['task_id'])


if __name__ == '__main__':
    try:
        validate(sys.argv[1], sys.argv[2])
    except Exception as exc:
        sys.exit(str(exc))
    print('ok: policy artifact; private observations are released only by scorer')
