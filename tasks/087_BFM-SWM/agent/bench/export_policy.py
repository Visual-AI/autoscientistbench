"""Seal code/ and the trained model into a policy artifact; no instance is accessed here."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import export_artifact

TASK_ID = "icml-4454-BFM-SWM"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("code", "model", "input", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    if request.get("task") != TASK_ID or request.get("protocol") != "private-policy-v2":
        raise ValueError("wrong public request descriptor")
    export_artifact(args.code, args.model, args.output, TASK_ID)
