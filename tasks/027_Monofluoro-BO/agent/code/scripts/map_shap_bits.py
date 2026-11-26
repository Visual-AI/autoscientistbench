#!/usr/bin/env python3
"""Map ranked component-specific Morgan bits to representative environments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_shap_analysis import map_ranked_bits, plot_bit_mappings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Reaction CSV containing sub_smiles1, sub_smiles2, and prod_smiles.",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Ranked-feature CSV produced by run_shap_analysis.py.",
    )
    parser.add_argument("--output", required=True, help="Output mapping CSV.")
    parser.add_argument("--figure", help="Optional highlighted-structure PNG.")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--top-n", type=int, default=4, help="Features per task.")
    parser.add_argument("--max-examples", type=int, default=3)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    ranked = pd.read_csv(args.features)
    if "task" not in ranked.columns:
        raise ValueError("The ranked-feature CSV must contain a task column.")
    selected = (
        ranked.sort_values(["task", "rank"])
        .groupby("task", sort=False, as_index=False)
        .head(args.top_n)
    )
    mappings = map_ranked_bits(
        data,
        selected,
        radius=args.radius,
        n_bits=args.n_bits,
        max_examples=args.max_examples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mappings.to_csv(output, index=False)
    if args.figure:
        plot_bit_mappings(mappings, Path(args.figure))
    print(f"Saved {len(mappings)} representative mappings to {output}")


if __name__ == "__main__":
    main()
