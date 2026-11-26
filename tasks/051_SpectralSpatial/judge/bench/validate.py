#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import expected_keys, read_predictions

if __name__ == '__main__':
    try:
        pred = read_predictions(sys.argv[1])
        expected = expected_keys(sys.argv[2])
        if set(pred) != expected:
            raise ValueError('missing or extra prediction keys')
        print(f'ok: {len(pred)} effect summaries')
    except Exception as exc:
        sys.exit(str(exc))
