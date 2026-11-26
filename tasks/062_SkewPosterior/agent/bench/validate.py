#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, rows, predictions

def main():
    try:
        expected = rows(load(sys.argv[2]))
        predictions(load(sys.argv[1]), expected)
    except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"valid": True, "cases": len(expected)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
