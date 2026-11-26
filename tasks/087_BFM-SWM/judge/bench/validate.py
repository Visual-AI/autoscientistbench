#!/usr/bin/env python3
"""Validate a policy artifact without executing candidate code."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact

TASK_ID = "icml-4454-BFM-SWM"


def main(pred_path, input_path):
    request = json.loads(Path(input_path).read_text())
    if request.get("task") != TASK_ID or request.get("protocol") != "private-policy-v2":
        raise ValueError("wrong public request descriptor")
    validate_artifact(pred_path, TASK_ID)
    print("ok: policy artifact; outcomes come only from the trusted offer oracle")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except Exception as exc:
        sys.exit(str(exc))
