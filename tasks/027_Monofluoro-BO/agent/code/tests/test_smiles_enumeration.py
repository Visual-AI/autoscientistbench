from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

from reaction_discovery.prediction.smiles_enumeration import (
    MOLECULE_COLUMNS,
    enumerate_split_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sub_smiles1": ["CCO", "CC(=O)C"],
            "sub_smiles2": ["OB(O)c1ccccc1", "CC1(C)OB(c2ccccc2)OC1(C)C"],
            "prod_smiles": ["CCOc1ccccc1", "CC(=O)c1ccccc1"],
            "yield": [25.0, 0.0],
            "cls_label": [1, 0],
            "split": ["train", "test"],
        }
    )


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True)


def test_each_reaction_keeps_original_and_adds_ten_paired_enumerations():
    source = sample_data()

    result = enumerate_split_dataframe(source)

    assert len(result) == 22
    assert result.groupby("source_row").size().to_dict() == {0: 11, 1: 11}
    for source_row, group in result.groupby("source_row"):
        assert group["enumeration_index"].tolist() == list(range(11))
        assert group["is_original"].tolist() == [True] + [False] * 10
        original = source.loc[source_row]
        assert group["split"].nunique() == 1
        assert group["split"].iloc[0] == original["split"]
        assert group["yield"].eq(original["yield"]).all()
        assert group["cls_label"].eq(original["cls_label"]).all()
        for column in MOLECULE_COLUMNS:
            assert group[column].map(canonical).eq(canonical(original[column])).all()
            assert group.iloc[0][column] == original[column]


def test_rejects_missing_or_invalid_split_before_enumeration():
    missing = sample_data().drop(columns="split")
    try:
        enumerate_split_dataframe(missing, enumerations=1)
    except ValueError as exc:
        assert "split" in str(exc)
    else:
        raise AssertionError("Missing split column was accepted")

    invalid = sample_data()
    invalid.loc[1, "split"] = "validation"
    try:
        enumerate_split_dataframe(invalid, enumerations=1)
    except ValueError as exc:
        assert "train" in str(exc) and "test" in str(exc)
    else:
        raise AssertionError("Unexpected split label was accepted")


def test_invalid_smiles_error_identifies_row_and_column():
    source = sample_data()
    source.loc[1, "sub_smiles2"] = "not-a-smiles"

    try:
        enumerate_split_dataframe(source, enumerations=1)
    except ValueError as exc:
        message = str(exc)
        assert "source row 1" in message
        assert "sub_smiles2" in message
    else:
        raise AssertionError("Invalid SMILES was accepted")


def test_cli_writes_expected_rows_without_seed_interface(tmp_path):
    input_path = tmp_path / "split.csv"
    output_path = tmp_path / "enumerated.csv"
    sample_data().to_csv(input_path, index=False)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_smiles_enumeration.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--enumerations",
        "2",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    output = pd.read_csv(output_path)
    assert len(output) == 6
    assert output.groupby("split").size().to_dict() == {"test": 3, "train": 3}
    assert "source reactions: 2" in completed.stdout
    assert "output rows: 6" in completed.stdout
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_smiles_enumeration.py"), "--help"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    assert "seed" not in help_text
