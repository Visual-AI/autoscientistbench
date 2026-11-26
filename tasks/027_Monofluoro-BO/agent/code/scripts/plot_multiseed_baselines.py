#!/usr/bin/env python3
"""Create Fig. 2D from the fair-pool multi-seed baseline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "Bayesian optimization": "#4F79A7",
    "Random search": "#B8C0CC",
    "Stratified random search": "#D99A3D",
}
EDGES = {
    "Bayesian optimization": "#274B70",
    "Random search": "#7D8794",
    "Stratified random search": "#9C641A",
}
METHODS = {
    "random": "Random search",
    "stratified_random": "Stratified random search",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default="data/baselines/processed/multiseed_aggregate_metrics.csv",
    )
    parser.add_argument(
        "--trajectory",
        default="data/baselines/raw/bo_trajectory.csv",
    )
    parser.add_argument(
        "--candidates",
        default="data/baselines/raw/measured_conditions.csv",
    )
    parser.add_argument(
        "--output",
        default="figures/baselines/fig2d_multiseed_baselines.png",
    )
    return parser.parse_args()


def boxed_axis(axis) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.05)
    axis.grid(False)
    axis.tick_params(direction="out", length=4, width=0.9, top=False, right=False)


def metric_rows(aggregate: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    rows = aggregate.loc[
        (aggregate["pool_definition"] == "fair")
        & (aggregate["method"] == method)
        & (aggregate["metric"] == metric)
    ].sort_values("budget")
    if len(rows) != 3:
        raise ValueError(f"Expected three budgets for {method}/{metric}, found {len(rows)}.")
    return rows


def main() -> None:
    args = parse_args()
    aggregate = pd.read_csv(args.aggregate)
    trajectory = pd.read_csv(args.trajectory)
    candidates = pd.read_csv(args.candidates, usecols=["data_index", "yield_percent"])
    trajectory = trajectory.drop(columns=["observed_y"], errors="ignore").merge(
        candidates, on="data_index", how="left", validate="one_to_one"
    )
    if trajectory["yield_percent"].isna().any():
        raise ValueError("One or more BO trajectory indices are absent from the candidate table.")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.65),
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.34},
    )

    budgets = np.array([5, 29, 53])
    bo_best = np.array(
        [trajectory.loc[trajectory["round"] <= stage, "yield_percent"].max() for stage in (0, 1, 2)]
    )
    left.plot(
        budgets,
        bo_best,
        color=COLORS["Bayesian optimization"],
        marker="o",
        markeredgecolor=EDGES["Bayesian optimization"],
        linewidth=2.2,
        markersize=6.5,
        label="Bayesian optimization",
        zorder=4,
    )
    for method, marker in (("random", "s"), ("stratified_random", "^")):
        label = METHODS[method]
        rows = metric_rows(aggregate, method, "best_yield")
        means = rows["mean"].to_numpy()
        lower = means - rows["q025"].to_numpy()
        upper = rows["q975"].to_numpy() - means
        left.errorbar(
            budgets,
            means,
            yerr=np.vstack([lower, upper]),
            color=COLORS[label],
            ecolor=EDGES[label],
            marker=marker,
            markeredgecolor=EDGES[label],
            markerfacecolor=COLORS[label],
            capsize=3.5,
            linewidth=2.0,
            elinewidth=1.25,
            markersize=6.5,
            label=label,
            zorder=3,
        )
    left.set_title("Cumulative best yield")
    left.set_xlabel("Number of selected conditions")
    left.set_ylabel("Best observed yield (%)")
    left.set_xticks(budgets)
    left.set_xlim(2, 57)
    left.set_ylim(0, 72)
    left.legend(frameon=False, loc="lower right", fontsize=8.5)
    boxed_axis(left)

    thresholds = (0, 10, 30, 50)
    x = np.arange(len(thresholds))
    width = 0.24
    bo_counts = np.array([(trajectory["yield_percent"] > threshold).sum() for threshold in thresholds])
    right.bar(
        x - width,
        bo_counts,
        width,
        color=COLORS["Bayesian optimization"],
        edgecolor=EDGES["Bayesian optimization"],
        linewidth=0.8,
        label="Bayesian optimization",
        zorder=3,
    )
    for offset, method in ((0, "random"), (width, "stratified_random")):
        label = METHODS[method]
        rows = []
        for threshold in thresholds:
            row = aggregate.loc[
                (aggregate["pool_definition"] == "fair")
                & (aggregate["method"] == method)
                & (aggregate["budget"] == 53)
                & (aggregate["metric"] == f"count_gt_{threshold}")
            ]
            if len(row) != 1:
                raise ValueError(f"Missing final hit-count summary for {method}, >{threshold}%.")
            rows.append(row.iloc[0])
        means = np.array([row["mean"] for row in rows])
        q025 = np.array([row["q025"] for row in rows])
        q975 = np.array([row["q975"] for row in rows])
        right.bar(
            x + offset,
            means,
            width,
            yerr=np.vstack([means - q025, q975 - means]),
            capsize=3,
            color=COLORS[label],
            edgecolor=EDGES[label],
            ecolor=EDGES[label],
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for xpos, value in zip(x + offset, means):
            right.text(
                xpos,
                max(0.15, value - 0.45),
                f"{value:.1f}",
                ha="center",
                va="top",
                fontsize=7.5,
                zorder=6,
                bbox={"facecolor": COLORS[label], "edgecolor": "none", "pad": 0.15},
            )
    for xpos, value in zip(x - width, bo_counts):
        right.text(xpos, value + 1.0, f"{value:d}", ha="center", va="bottom", fontsize=8)
    right.set_title("Hit enrichment at the 53-condition budget")
    right.set_xlabel("Yield threshold")
    right.set_ylabel("Number of conditions among 53")
    right.set_xticks(x, [f">{threshold}%" for threshold in thresholds])
    right.set_ylim(0, 39)
    boxed_axis(right)
    right.text(
        0.98,
        0.98,
        "Baselines: mean and empirical 95% interval\nn = 50 seeds; fair pool = 371",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )

    fig.text(0.012, 0.985, "D", fontsize=18, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.86, wspace=0.34)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Saved {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
