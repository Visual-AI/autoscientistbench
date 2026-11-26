#!/usr/bin/env python3
"""validate.py PRED INPUT: exit 0 only if PRED is a well-formed purchase family for the issues in INPUT."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import FamilyError, load_family, read_issue_ids  # noqa: E402


def main(pred_path, input_path):
    ids = read_issue_ids(input_path)
    family = load_family(pred_path, ids)
    print(f"ok: {len(family)} levels x {len(ids)} issues")


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except (FamilyError, ValueError, KeyError, OSError) as exc:
        sys.exit(f"invalid: {exc}")
