#!/usr/bin/env python3
"""Protocol entry points around the authors' analysis of Figure 2 (notebooks/swebench_gpt5_deepseek.ipynb, cells 2, 5, 7).

train:   the notebook's grid search for the cascade allocation tau, by mean profit margin, on data/train only.
predict: Eq. 7 of the paper at each point of the notebook's budget grid, written as a purchase family.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "notebooks"))

MODELS = ["gpt5mini", "deepseek"]


def budget_levels():
    return np.logspace(np.log10(1e-4), np.log10(1), 250)  # notebook cell 5


def fit_tau(results):
    """Notebook cells 5 and 7, unchanged."""
    import jax.numpy as jnp
    from utils import load_pass_curves, threshold_to_profit_batch

    budget_grid = budget_levels()
    providers_perf = load_pass_curves(results, MODELS, budget_grid)
    max_perf = providers_perf.mean(axis=0).max()
    performance_grid = jnp.linspace(0.68, max_perf, 1000)
    compute_first = np.r_[0, np.logspace(np.log10(0.01), np.log10(0.5), 100), budget_grid.max()]
    compute_second = budget_grid.max() - compute_first
    compute_allocation = np.stack([compute_first, compute_second], axis=-1)
    mean_profits = threshold_to_profit_batch(
        budget_grid, providers_perf, compute_allocation, performance_grid
    ).mean(axis=-1)
    return compute_allocation[np.argmax(mean_profits, axis=-1)]


def segments(tau, budget):
    """Eq. 7: provider i receives min(max(b - sum_{k<i} tau_k, 0), tau_i)."""
    spent_before, out = 0.0, []
    for model, cap in zip(MODELS, tau):
        dollars = min(max(budget - spent_before, 0.0), cap)
        if dollars > 0:
            out.append((model, float(dollars)))
        spent_before += cap
    return out


def train(out_dir):
    train_dir = Path(os.environ["BENCH_DATA_DIR"]) / "train"
    results = {m: [json.loads(line) for line in open(train_dir / f"{m}.jsonl")] for m in MODELS}
    tau = fit_tau(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tau.json").write_text(json.dumps({"models": MODELS, "tau": [float(t) for t in tau]}) + "\n")
    print("fitted tau:", [float(t) for t in tau])


def predict(model_dir, input_path, output_path):
    tau = json.loads((Path(model_dir) / "tau.json").read_text())["tau"]
    ids = [json.loads(line)["instance_id"] for line in open(input_path, encoding="utf-8") if line.strip()]
    with open(output_path, "w", encoding="ascii") as f:
        f.write("level,instance_id,plan\n")
        for level, budget in enumerate(budget_levels()):
            plan = ";".join(f"{m}:{d!r}" for m, d in segments(tau, float(budget)))
            for iid in ids:
                f.write(f"{level},{iid},{plan}\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.out)
    else:
        predict(a.model, a.input, a.output)


if __name__ == "__main__":
    main()
