#!/usr/bin/env python
"""Run organoboron reagent-pool curation and stratified sampling.

Example:
    PYTHONPATH=src python scripts/run_substrate_panel_selection.py \
        --raw-dir data/substrate_selection/raw \
        --output-dir data/substrate_selection/processed \
        --n-samples 38
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reaction_discovery.substrate_selection import (
    combined_pca_dataframe,
    curate_pool,
    extract_reactant_smiles_from_uspto,
    load_smiles_from_csv,
    load_smiles_from_excel,
    stratified_radial_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Chemspace/Topscience pools and run stratified sampling.")
    parser.add_argument("--raw-dir", default="data/substrate_selection/raw", help="Directory containing cpm*.xlsx, USPTO_50K.csv, and topscience_full.csv.")
    parser.add_argument("--output-dir", default="data/substrate_selection/processed", help="Directory for processed outputs.")
    parser.add_argument("--n-samples", type=int, default=38, help="Number of automatically cluster/strata-selected Topscience compounds.")
    parser.add_argument("--n-layers", type=int, default=10, help="Number of radial PCA strata for sampling.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint bit length.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpm_files = sorted(raw_dir.glob("cpm*.xlsx"))
    if not cpm_files:
        raise FileNotFoundError(f"No cpm*.xlsx files found in {raw_dir}")

    chemspace_raw: list[str] = []
    for path in cpm_files:
        chemspace_raw.extend(load_smiles_from_excel(path, smiles_col="canonical_smiles"))

    uspto_path = raw_dir / "USPTO_50K.csv"
    if uspto_path.exists():
        chemspace_raw.extend(extract_reactant_smiles_from_uspto(uspto_path))

    topscience_path = raw_dir / "topscience_full.csv"
    if not topscience_path.exists():
        raise FileNotFoundError(f"Cannot find {topscience_path}")
    topscience_raw = load_smiles_from_csv(topscience_path)

    chemspace_hits, chemspace_summary = curate_pool("Chemspace", chemspace_raw)
    topscience_hits, topscience_summary = curate_pool("Topscience", topscience_raw)

    pd.DataFrame({"smiles": chemspace_hits}).to_csv(out_dir / "chemspace_hits.csv", index=False)
    pd.DataFrame({"smiles": topscience_hits}).to_csv(out_dir / "topscience_hits.csv", index=False)

    summary_df = pd.DataFrame([chemspace_summary.__dict__, topscience_summary.__dict__])
    summary_df.to_csv(out_dir / "pool_summary.csv", index=False)

    sample_df = stratified_radial_sample(
        topscience_hits,
        n_samples=args.n_samples,
        n_layers=args.n_layers,
        radius=args.radius,
        n_bits=args.n_bits,
        random_state=args.random_state,
    )
    sample_df.to_csv(out_dir / "topscience_stratified_sampling.csv", index=False)
    sampled_smiles = sample_df.loc[sample_df["selected"], "smiles"].tolist()
    pd.DataFrame({"smiles": sampled_smiles}).to_csv(out_dir / "sampled_hits.csv", index=False)

    pca_df = combined_pca_dataframe(
        chemspace_hits,
        topscience_hits,
        sampled_smiles,
        radius=args.radius,
        n_bits=args.n_bits,
        random_state=args.random_state,
    )
    pca_df.to_csv(out_dir / "pca_coordinates.csv", index=False)

    colors = {"Chemspace": "#1f77b4", "Topscience": "#f2a900", "Sampled": "#d62728"}
    markers = {"Chemspace": "o", "Topscience": "^", "Sampled": "*"}
    plt.figure(figsize=(6, 6), dpi=300)
    for name in ["Chemspace", "Topscience", "Sampled"]:
        sub = pca_df[pca_df["dataset"] == name]
        if sub.empty:
            continue
        size = 50 if name == "Sampled" else 8
        alpha = 0.9 if name == "Sampled" else 0.6
        plt.scatter(sub["PC1"], sub["PC2"], s=size, alpha=alpha, c=colors[name], marker=markers[name], edgecolors="black", linewidths=0.1, label=name)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_scatter.png", dpi=300)
    plt.close()

    print("Pool summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved processed outputs to {out_dir}")


if __name__ == "__main__":
    main()
