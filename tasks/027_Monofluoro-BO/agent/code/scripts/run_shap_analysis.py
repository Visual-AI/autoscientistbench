#!/usr/bin/env python3
"""Reproduce fixed-split XGBoost SHAP summaries and bit mappings."""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

from reaction_discovery.prediction.staged_model_cli import featurize_dataframe


BLOCK_NAMES = ("substrate1", "organoboron", "product")
BLOCK_COLUMNS = {
    "substrate1": "sub_smiles1",
    "organoboron": "sub_smiles2",
    "product": "prod_smiles",
}


def decode_feature_index(feature_index: int, bits: int) -> tuple[str, int]:
    """Return the component block and local Morgan bit for a global index."""
    if feature_index < 0 or feature_index >= 3 * bits:
        raise ValueError(f"Feature index {feature_index} is outside 0..{3 * bits - 1}.")
    block_index, local_bit = divmod(int(feature_index), bits)
    return BLOCK_NAMES[block_index], local_bit


def rank_mean_absolute_shap(
    shap_values: np.ndarray, bits: int, top_n: int
) -> pd.DataFrame:
    """Rank composite fingerprint features by mean absolute SHAP value."""
    values = np.asarray(shap_values)
    if values.ndim != 2 or values.shape[1] != 3 * bits:
        raise ValueError(
            f"Expected a two-dimensional SHAP matrix with {3 * bits} columns; "
            f"received {values.shape}."
        )
    scores = np.abs(values).mean(axis=0)
    ranked_indices = np.argsort(scores, kind="stable")[::-1][:top_n]
    rows = []
    for rank, feature_index in enumerate(ranked_indices, start=1):
        block, local_bit = decode_feature_index(int(feature_index), bits)
        rows.append(
            {
                "rank": rank,
                "global_feature": int(feature_index),
                "feature_block": block,
                "bit": local_bit,
                "mean_abs_shap": float(scores[feature_index]),
            }
        )
    return pd.DataFrame(rows)


def _atom_environment(
    mol: Chem.Mol, center_atom: int, radius: int
) -> tuple[str, list[int]]:
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom))
    atoms = {center_atom}
    for bond_index in bonds:
        bond = mol.GetBondWithIdx(bond_index)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    atom_list = sorted(atoms)
    environment_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atom_list,
        bondsToUse=bonds,
        canonical=True,
    )
    return environment_smiles, atom_list


def map_ranked_bits(
    data: pd.DataFrame,
    ranked_features: pd.DataFrame,
    radius: int,
    n_bits: int,
    max_examples: int = 1,
) -> pd.DataFrame:
    """Map ranked bits to representative local environments in the right block."""
    required = set(BLOCK_COLUMNS.values())
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing molecular columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    RDLogger.DisableLog("rdApp.warning")
    try:
        for feature in ranked_features.to_dict("records"):
            block = str(feature["feature_block"])
            source_column = BLOCK_COLUMNS[block]
            bit = int(feature["bit"])
            occurrences: list[dict[str, object]] = []
            parent_indices: set[int] = set()
            parent_smiles_values: set[str] = set()
            environments: set[str] = set()
            for parent_index, smiles in data[source_column].dropna().items():
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                bit_info: dict[int, list[tuple[int, int]]] = {}
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius, nBits=n_bits, bitInfo=bit_info
                )
                for center_atom, environment_radius in bit_info.get(bit, []):
                    parent_smiles = Chem.MolToSmiles(mol, canonical=True)
                    environment_smiles, highlight_atoms = _atom_environment(
                        mol, center_atom, environment_radius
                    )
                    parent_indices.add(int(parent_index))
                    parent_smiles_values.add(parent_smiles)
                    environments.add(environment_smiles)
                    occurrences.append(
                        {
                            "parent_index": int(parent_index),
                            "parent_smiles": parent_smiles,
                            "center_atom": int(center_atom),
                            "environment_radius": int(environment_radius),
                            "environment_smiles": environment_smiles,
                            "highlight_atoms": ",".join(map(str, highlight_atoms)),
                        }
                    )

            seen_environments: set[str] = set()
            examples = []
            for occurrence in occurrences:
                environment = str(occurrence["environment_smiles"])
                if environment in seen_environments:
                    continue
                seen_environments.add(environment)
                examples.append(occurrence)
                if len(examples) == max_examples:
                    break

            for example_number, occurrence in enumerate(examples, start=1):
                rows.append(
                    {
                        **feature,
                        "source_column": source_column,
                        "example": example_number,
                        "parent_molecule_count": len(parent_smiles_values),
                        "parent_row_count": len(parent_indices),
                        "environment_variant_count": len(environments),
                        **occurrence,
                    }
                )
    finally:
        RDLogger.EnableLog("rdApp.warning")
    return pd.DataFrame(rows)


def _normalise_shap_output(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def tree_shap_values(model: object, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exact TreeSHAP contributions with XGBoost's native predictor."""
    import xgboost as xgb

    contributions = np.asarray(
        model.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)
    )
    if contributions.ndim != 2 or contributions.shape[1] != features.shape[1] + 1:
        raise ValueError(f"Unexpected TreeSHAP contribution shape: {contributions.shape}")
    return contributions[:, :-1], contributions[:, -1]


def aggregate_blocks(shap_values: np.ndarray, bits: int) -> dict[str, dict[str, float]]:
    totals = {
        name: float(np.abs(shap_values[:, index * bits : (index + 1) * bits]).sum())
        for index, name in enumerate(BLOCK_NAMES)
    }
    total = sum(totals.values())
    return {
        name: {"sum_abs_shap": value, "fraction": value / total}
        for name, value in totals.items()
    }


def plot_shap_summary(
    summary: dict[str, object],
    classifier_ranked: pd.DataFrame,
    regressor_ranked: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(12, 8), dpi=300)
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 2), hspace=0.35, wspace=0.28)
    ax_block = fig.add_subplot(grid[0, :])
    ax_classifier = fig.add_subplot(grid[1, 0])
    ax_regressor = fig.add_subplot(grid[1, 1])

    x = np.arange(len(BLOCK_NAMES))
    width = 0.35
    classifier_fractions = [summary["classifier"][name]["fraction"] for name in BLOCK_NAMES]
    regressor_fractions = [summary["regressor"][name]["fraction"] for name in BLOCK_NAMES]
    classifier_bars = ax_block.bar(
        x - width / 2, classifier_fractions, width, label="Classifier", color="#4C78A8"
    )
    regressor_bars = ax_block.bar(
        x + width / 2, regressor_fractions, width, label="Regressor", color="#F58518"
    )
    ax_block.bar_label(classifier_bars, fmt="%.2f", padding=2, fontsize=8)
    ax_block.bar_label(regressor_bars, fmt="%.2f", padding=2, fontsize=8)
    ax_block.set_xticks(x, ["Substrate 1", "Organoboron reagent", "Product"])
    ax_block.set_ylabel("Fraction of total mean |SHAP|")
    ax_block.set_title("A  Feature-block contributions", loc="left")
    ax_block.legend(frameon=False)

    for axis, ranked, title, color in (
        (ax_classifier, classifier_ranked, "B  Classifier: top fingerprint features", "#4C78A8"),
        (ax_regressor, regressor_ranked, "C  Regressor: top fingerprint features", "#F58518"),
    ):
        labels = [f"{row.feature_block}:{row.bit}" for row in ranked.itertuples()]
        axis.barh(labels[::-1], ranked["mean_abs_shap"].to_numpy()[::-1], color=color)
        axis.set_xlabel("Mean |SHAP value|")
        axis.set_title(title, loc="left")
        axis.tick_params(axis="y", labelsize=7)

    fig.suptitle("SHAP interpretation of the fixed two-stage XGBoost workflow", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.13, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_bit_mappings(mappings: pd.DataFrame, output: Path, max_panels: int = 8) -> None:
    selected = mappings[mappings["example"] == 1].head(max_panels)
    mols = []
    legends = []
    highlights = []
    for row in selected.itertuples():
        mol = Chem.MolFromSmiles(row.parent_smiles)
        if mol is None:
            continue
        mols.append(mol)
        highlights.append([int(value) for value in str(row.highlight_atoms).split(",") if value])
        legends.append(
            f"{str(row.task).capitalize()} {row.feature_block}:{row.bit}\n"
            f"mean |SHAP| = {row.mean_abs_shap:.3f}"
        )
    if not mols:
        raise RuntimeError("No representative bit environments were found.")
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(520, 400),
        legends=legends,
        highlightAtomLists=highlights,
        useSVG=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prediction/raw/dataset_split.csv")
    parser.add_argument("--model-dir", default="models/prediction")
    parser.add_argument("--output-dir", default="data/interpretation/shap")
    parser.add_argument("--figure-dir", default="figures/interpretation/shap")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--mapping-top-n", type=int, default=4)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    test = data[data["split"].astype(str).str.lower().eq("test")].reset_index(drop=True)
    X, y, kept_test = featurize_dataframe(
        test, bits=args.n_bits, radius=args.radius, require_yield=True
    )
    with open(Path(args.model_dir) / "xgboost_classifier.pkl", "rb") as handle:
        classifier = pickle.load(handle)
    with open(Path(args.model_dir) / "xgboost_regressor.pkl", "rb") as handle:
        regressor = pickle.load(handle)

    classifier_values, _ = tree_shap_values(classifier, X)
    productive = y > 0
    regressor_values, _ = tree_shap_values(regressor, X[productive])

    summary = {
        "split": {
            "train": int((data["split"].astype(str).str.lower() == "train").sum()),
            "test": len(kept_test),
            "productive_test": int(productive.sum()),
        },
        "fingerprint": {"radius": args.radius, "n_bits_per_component": args.n_bits},
        "classifier": aggregate_blocks(classifier_values, args.n_bits),
        "regressor": aggregate_blocks(regressor_values, args.n_bits),
    }
    classifier_ranked = rank_mean_absolute_shap(
        classifier_values, args.n_bits, args.top_n
    )
    classifier_ranked.insert(0, "task", "classifier")
    regressor_ranked = rank_mean_absolute_shap(
        regressor_values, args.n_bits, args.top_n
    )
    regressor_ranked.insert(0, "task", "regressor")
    ranked = pd.concat([classifier_ranked, regressor_ranked], ignore_index=True)
    mapping_candidates = pd.concat(
        [
            classifier_ranked.head(args.mapping_top_n),
            regressor_ranked.head(args.mapping_top_n),
        ],
        ignore_index=True,
    )
    mappings = map_ranked_bits(
        data,
        mapping_candidates,
        radius=args.radius,
        n_bits=args.n_bits,
        max_examples=3,
    )

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shap_two_stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "feature_block": BLOCK_NAMES,
            "classifier_fraction": [summary["classifier"][k]["fraction"] for k in BLOCK_NAMES],
            "regressor_fraction": [summary["regressor"][k]["fraction"] for k in BLOCK_NAMES],
        }
    ).to_csv(output_dir / "shap_block_contributions.csv", index=False)
    ranked.to_csv(output_dir / "shap_top_features.csv", index=False)
    mappings.to_csv(output_dir / "shap_bit_mappings.csv", index=False)

    summary_figure = output_dir / "shap_two_stage_summary.png"
    mapping_figure = output_dir / "shap_bit_mappings.png"
    plot_shap_summary(summary, classifier_ranked, regressor_ranked, summary_figure)
    plot_bit_mappings(mappings, mapping_figure)
    shutil.copy2(summary_figure, figure_dir / summary_figure.name)
    shutil.copy2(mapping_figure, figure_dir / mapping_figure.name)
    print(json.dumps(summary, indent=2))
    print(f"Saved feature rankings and {len(mappings)} representative mappings to {output_dir}")


if __name__ == "__main__":
    main()
