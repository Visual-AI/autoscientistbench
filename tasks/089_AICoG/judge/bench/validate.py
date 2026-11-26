#!/usr/bin/env python3
"""python validate.py PRED INPUT -> exit 0 if PRED is a well-formed representation for the node count in INPUT.

Uses the scorer's own loader, so a file passes here exactly when the scorer accepts its format.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import load_prediction  # noqa: E402


def main(pred_path, input_path):
    with open(input_path, encoding="utf-8") as f:
        n_nodes = int(json.load(f)["n_nodes"])
    z, _, name = load_prediction(pred_path, n_nodes)
    print(f"ok: z {z.shape}, decoder {name}")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:3])
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"invalid: {type(exc).__name__}: {exc}"[:500])
