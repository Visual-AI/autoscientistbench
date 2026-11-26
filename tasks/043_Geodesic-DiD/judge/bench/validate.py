import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_tools import input_ids, read_predictions

if __name__ == '__main__':
    try:
        read_predictions(sys.argv[1], input_ids(sys.argv[2]))
    except Exception as exc:
        print(f'invalid: {exc}', file=sys.stderr)
        raise SystemExit(1)
    print('valid')
