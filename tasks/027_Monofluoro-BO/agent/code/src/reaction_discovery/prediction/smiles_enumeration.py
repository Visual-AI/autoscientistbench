"""Post-split randomized SMILES enumeration for reaction datasets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


MOLECULE_COLUMNS = ("sub_smiles1", "sub_smiles2", "prod_smiles")
PROVENANCE_COLUMNS = ("source_row", "enumeration_index", "is_original")


def _validate_input(data: pd.DataFrame, enumerations: int) -> pd.DataFrame:
    if not isinstance(enumerations, int) or isinstance(enumerations, bool):
        raise TypeError("enumerations must be an integer")
    if enumerations < 0:
        raise ValueError("enumerations must be non-negative")

    required = {*MOLECULE_COLUMNS, "split"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    conflicts = sorted(set(PROVENANCE_COLUMNS).intersection(data.columns))
    if conflicts:
        raise ValueError(
            "Input already contains reserved provenance columns: "
            + ", ".join(conflicts)
        )

    result = data.copy().reset_index(drop=True)
    if result["split"].isna().any():
        raise ValueError("split must be populated with train or test for every row")
    normalized_split = result["split"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized_split).difference({"train", "test"}))
    if unexpected:
        raise ValueError(
            "split must contain only train and test; found: " + ", ".join(unexpected)
        )
    if set(normalized_split) != {"train", "test"}:
        raise ValueError("split must contain both train and test subsets")
    result["split"] = normalized_split
    return result


def _parse_molecules(row: Mapping[str, object], source_row: int) -> dict[str, Chem.Mol]:
    molecules: dict[str, Chem.Mol] = {}
    for column in MOLECULE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            raise ValueError(f"Invalid SMILES at source row {source_row}, column {column}: missing")
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(
                f"Invalid SMILES at source row {source_row}, column {column}: {value!r}"
            )
        molecules[column] = molecule
    return molecules


def _random_smiles(molecule: Chem.Mol, rng: np.random.Generator) -> str:
    """Return one valid non-canonical SMILES from randomized atom traversal."""
    atom_order = rng.permutation(molecule.GetNumAtoms()).tolist()
    renumbered = Chem.RenumberAtoms(molecule, atom_order)
    return Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True)


def enumerate_split_dataframe(
    data: pd.DataFrame, enumerations: int = 10
) -> pd.DataFrame:
    """Keep each source row and add paired randomized SMILES representations.

    The reaction-level split must already exist. Each output row inherits the
    source row's metadata and subset assignment. Molecular columns are
    randomized independently for each shared enumeration index; no Cartesian
    product is formed.
    """
    source = _validate_input(data, enumerations)
    output_rows: list[dict[str, object]] = []
    rng = np.random.default_rng()

    RDLogger.DisableLog("rdApp.error")
    try:
        for source_row, row in source.iterrows():
            values = row.to_dict()
            molecules = _parse_molecules(values, source_row)
            output_rows.append(
                {
                    **values,
                    "source_row": source_row,
                    "enumeration_index": 0,
                    "is_original": True,
                }
            )
            for enumeration_index in range(1, enumerations + 1):
                randomized = {
                    column: _random_smiles(molecule, rng)
                    for column, molecule in molecules.items()
                }
                output_rows.append(
                    {
                        **values,
                        **randomized,
                        "source_row": source_row,
                        "enumeration_index": enumeration_index,
                        "is_original": False,
                    }
                )
    finally:
        RDLogger.EnableLog("rdApp.error")

    return pd.DataFrame(output_rows, columns=[*source.columns, *PROVENANCE_COLUMNS])
