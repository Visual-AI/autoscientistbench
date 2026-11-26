#!/usr/bin/env python3
"""Run the manuscript's matched-budget retrospective baseline analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaction_discovery.baseline_selection import (
    normalize_candidates,
    normalize_trajectory,
    read_table,
    run_many_seeds,
    summarize_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="data/baselines/raw/measured_conditions.csv",
        help="376-row measured candidate table (CSV or Excel)",
    )
    parser.add_argument(
        "--trajectory",
        default="data/baselines/raw/bo_trajectory.csv",
        help="53-row Bayesian-optimization trajectory (CSV or Excel)",
    )
    parser.add_argument("--output-dir", default="data/baselines/processed")
    parser.add_argument("--seeds", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = normalize_candidates(read_table(args.candidates))
    trajectory = normalize_trajectory(read_table(args.trajectory))
    selections = run_many_seeds(candidates, trajectory, seeds=range(args.seeds))
    per_seed, aggregate = summarize_runs(selections)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection_columns = [
        "pool_definition",
        "seed",
        "method",
        "stage",
        "selection_order",
        "data_index",
        "Cat.",
        "base",
        "Sol.",
        "Temp./C",
        "cat_family",
        "base_class",
        "solvent_class",
        "temp_class",
        "yield_percent",
    ]
    selections[selection_columns].to_csv(output / "multiseed_selections.csv", index=False)
    per_seed.to_csv(output / "multiseed_per_seed_metrics.csv", index=False)
    aggregate.to_csv(output / "multiseed_aggregate_metrics.csv", index=False)
    selections.loc[selections["seed"] == 42, selection_columns].to_csv(
        output / "seed_42_selections.csv", index=False
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
