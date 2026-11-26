#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expressions import load, read_table, validate_prediction

if __name__ == "__main__":
    try:
        rows = read_table(sys.argv[2])
        validate_prediction(load(sys.argv[1]), rows)
        print(f"ok: {len(rows)} certified rows")
    except Exception as exc:
        raise SystemExit(f"invalid prediction: {exc}")
