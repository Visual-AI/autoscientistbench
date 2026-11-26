"""Utilities for organoboron reagent pool construction and stratified sampling.

This module packages the substrate/reagent-panel processing workflow used for
Chemspace/USPTO/TopScience pool curation, fingerprint-based projection, and
stratified sampling of purchasable organoboron reagents.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA


BORON_SMARTS = [
    "[#6]B1OC(C)(C)C(C)(C)O1",   # boronic ester / Bpin-like fragment
    "[#6]B(O)O",                 # boronic acid-like fragment
    "[#6]B1OB([#6])OB([#6])O1",  # boronic anhydride-like fragment
]


@dataclass
class PoolSummary:
    """Before/after statistics for a curated SMILES pool."""

    name: str
    raw_records: int
    valid_smiles: int
    matched_records: int
    unique_matched: int


# -----------------------------------------------------------------------------
# Basic chemistry helpers
# -----------------------------------------------------------------------------
def canonicalize_smiles(smiles: str | float | None) -> str | None:
    """Return canonical SMILES or None if parsing fails."""
    if smiles is None or pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_smiles_from_csv(path: str | Path, smiles_col: str | None = None) -> list[str]:
    """Load a SMILES column from a CSV file.

    If `smiles_col` is not provided, the first matching column among common SMILES
    names is used. As a last resort, the first column is used.
    """
    df = pd.read_csv(path, on_bad_lines="skip")
    if smiles_col is None:
        candidates = ["smiles", "SMILES", "can_smiles_from_mol", "canonical_smiles", "mol"]
        smiles_col = next((c for c in candidates if c in df.columns), df.columns[0])
    return df[smiles_col].dropna().astype(str).tolist()


def load_smiles_from_excel(path: str | Path, smiles_col: str = "canonical_smiles") -> list[str]:
    """Load a SMILES column from an Excel file."""
    df = pd.read_excel(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col!r} not found in {path}. Columns: {list(df.columns)}")
    return df[smiles_col].dropna().astype(str).tolist()


def extract_reactant_smiles_from_uspto(path: str | Path, reaction_col: str = "reactions") -> list[str]:
    """Extract individual reactant SMILES from a USPTO-style reaction CSV."""
    df = pd.read_csv(path)
    if reaction_col not in df.columns:
        raise ValueError(f"Column {reaction_col!r} not found in {path}. Columns: {list(df.columns)}")
    reactants: list[str] = []
    for reaction in df[reaction_col].dropna().astype(str):
        left = reaction.split(">>")[0]
        reactants.extend([s for s in left.split(".") if s])
    return reactants


def match_smarts(smiles_list: Sequence[str], smarts_list: Sequence[str] = BORON_SMARTS) -> list[str]:
    """Return canonical SMILES that match at least one SMARTS pattern."""
    patterns = [Chem.MolFromSmarts(s) for s in smarts_list]
    patterns = [p for p in patterns if p is not None]
    hits: list[str] = []
    for smiles in smiles_list:
        can = canonicalize_smiles(smiles)
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if any(mol.HasSubstructMatch(pat) for pat in patterns):
            hits.append(can)
    return hits


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Deduplicate while preserving first occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def curate_pool(name: str, smiles_list: Sequence[str], smarts_list: Sequence[str] = BORON_SMARTS) -> tuple[list[str], PoolSummary]:
    """Canonicalize, substructure-filter, and deduplicate a SMILES pool."""
    canonical = [canonicalize_smiles(s) for s in smiles_list]
    valid = [s for s in canonical if s is not None]
    matched = match_smarts(valid, smarts_list=smarts_list)
    unique = unique_preserve_order(matched)
    summary = PoolSummary(
        name=name,
        raw_records=len(smiles_list),
        valid_smiles=len(valid),
        matched_records=len(matched),
        unique_matched=len(unique),
    )
    return unique, summary


def morgan_fingerprint_matrix(smiles_list: Sequence[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute a Morgan fingerprint matrix for valid SMILES."""
    fps: list[np.ndarray] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    if not fps:
        return np.empty((0, n_bits), dtype=np.int8)
    return np.vstack(fps)


def stratified_radial_sample(
    smiles_list: Sequence[str],
    n_samples: int = 38,
    n_layers: int = 10,
    radius: int = 2,
    n_bits: int = 2048,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample a structurally diverse subset from a SMILES pool.

    Molecules are embedded by Morgan fingerprints followed by PCA. The radial
    distance in PC space is divided into quantile layers, and samples are drawn
    across layers to cover both dense central regions and peripheral structures.
    """
    rng = np.random.default_rng(random_state)
    clean_smiles = [s for s in smiles_list if canonicalize_smiles(s) is not None]
    fps = morgan_fingerprint_matrix(clean_smiles, radius=radius, n_bits=n_bits)
    if len(clean_smiles) == 0:
        raise ValueError("No valid SMILES available for sampling.")
    if len(clean_smiles) <= n_samples:
        return pd.DataFrame({"smiles": clean_smiles, "selected": True, "PC1": np.nan, "PC2": np.nan, "layer": 0})

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(fps)
    pool = pd.DataFrame({"smiles": clean_smiles, "PC1": coords[:, 0], "PC2": coords[:, 1]})
    pool["radial_distance"] = np.sqrt(pool["PC1"] ** 2 + pool["PC2"] ** 2)
    pool["layer"] = pd.qcut(pool["radial_distance"], q=min(n_layers, len(pool)), labels=False, duplicates="drop")

    selected_indices: list[int] = []
    layers = sorted(pool["layer"].dropna().unique())
    base_quota = max(1, n_samples // max(1, len(layers)))
    for layer in layers:
        layer_idx = pool.index[pool["layer"] == layer].to_numpy()
        if len(layer_idx) == 0:
            continue
        take = min(base_quota, len(layer_idx))
        selected_indices.extend(rng.choice(layer_idx, size=take, replace=False).tolist())

    remaining = n_samples - len(selected_indices)
    if remaining > 0:
        rest = np.array([i for i in pool.index if i not in set(selected_indices)])
        if len(rest) > 0:
            selected_indices.extend(rng.choice(rest, size=min(remaining, len(rest)), replace=False).tolist())

    selected_indices = selected_indices[:n_samples]
    pool["selected"] = False
    pool.loc[selected_indices, "selected"] = True
    return pool


def combined_pca_dataframe(
    chemspace_smiles: Sequence[str],
    topscience_smiles: Sequence[str],
    sampled_smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a combined PCA dataframe for Chemspace, Topscience, and sampled sets."""
    records: list[tuple[str, str]] = []
    for label, smiles_list in [
        ("Chemspace", chemspace_smiles),
        ("Topscience", topscience_smiles),
        ("Sampled", sampled_smiles),
    ]:
        for smiles in smiles_list:
            can = canonicalize_smiles(smiles)
            if can is not None:
                records.append((label, can))
    all_smiles = [r[1] for r in records]
    fps = morgan_fingerprint_matrix(all_smiles, radius=radius, n_bits=n_bits)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(fps)
    return pd.DataFrame({
        "dataset": [r[0] for r in records],
        "smiles": all_smiles,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })
