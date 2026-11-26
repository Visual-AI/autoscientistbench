"""The outer output is an inert policy archive; CSV is checked after replay."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact


def main(pred_path, input_path):
    request = json.loads(Path(input_path).read_text())
    if request.get('task') != 'Che-06' or request.get('protocol') != 'private-policy-v2':
        raise ValueError('wrong task request')
    validate_artifact(pred_path, 'Che-06')
    print('ok: inert policy archive; trusted replay checks complete finite predictions')


if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception:
        sys.exit('invalid Che-06 policy archive or request')
