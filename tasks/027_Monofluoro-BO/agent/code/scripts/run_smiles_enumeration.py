#!/usr/bin/env python3
"""Generate randomized SMILES representations after a fixed dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reaction_discovery.prediction.smiles_enumeration import enumerate_split_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate molecular SMILES after a fixed reaction-level split."
    )
    parser.add_argument(
        "--input",
        default="data/prediction/raw/dataset_split.csv",
        help="CSV containing molecular columns and an existing train/test split",
    )
    parser.add_argument(
        "--output",
        default="data/prediction/processed/dataset_split_enumerated.csv",
        help="Output CSV for original and enumerated representations",
    )
    parser.add_argument(
        "--enumerations",
        type=int,
        default=10,
        help="Number of randomized representations added per source reaction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    result = enumerate_split_dataframe(source, enumerations=args.enumerations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    counts = result.groupby("split").size().to_dict()
    print(f"source reactions: {len(source)}")
    print(f"output rows: {len(result)}")
    print(f"train rows: {counts.get('train', 0)}")
    print(f"test rows: {counts.get('test', 0)}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
