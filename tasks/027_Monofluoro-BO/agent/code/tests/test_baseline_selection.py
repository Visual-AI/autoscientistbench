from pathlib import Path

import numpy as np
import pandas as pd

from reaction_discovery.baseline_selection import (
    add_class_columns,
    build_candidate_pool,
    normalize_candidates,
    run_baseline_seed,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "baselines" / "raw" / "measured_conditions.csv"
TRAJECTORY = ROOT / "data" / "baselines" / "raw" / "bo_trajectory.csv"


def load_inputs():
    candidates = pd.read_csv(CANDIDATES)
    trajectory = pd.read_csv(TRAJECTORY)
    return candidates, trajectory


def test_classification_matches_documented_coarse_groups():
    rows = pd.DataFrame(
        {
            "Cat.": ["Pd168", "Cu27", "MgCl2"],
            "base": ["NaHCO3", "LiCl", "NMM"],
            "Sol.": ["1,4-dioxane", "DCE", "MeCN"],
            "Temp./C": [110, 80, 80],
        }
    )

    classified = add_class_columns(rows)

    assert classified["cat_family"].tolist() == ["Pd", "Cu", "MgCl"]
    assert classified["base_class"].tolist() == [
        "carbonate/bicarbonate",
        "halide",
        "amine",
    ]
    assert classified["solvent_class"].tolist() == ["ether", "halogenated", "nitrile"]
    assert classified["temp_class"].tolist() == ["110", "80", "80"]


def test_raw_excel_temperature_header_is_normalized_for_cli_output():
    rows = pd.DataFrame(
        {
            "Cat.": ["Pd168"],
            "base": ["KHCO3"],
            "Sol.": ["DCE"],
            "Temp./℃": [80],
            "Yield": [0.65],
        }
    )

    normalized = normalize_candidates(rows)

    assert "Temp./C" in normalized.columns
    assert "Temp./℃" not in normalized.columns
    assert normalized.loc[0, "yield_percent"] == 65.0


def test_fair_pool_excludes_only_shared_initialization():
    candidates, trajectory = load_inputs()

    pool, initial = build_candidate_pool(candidates, trajectory, pool_definition="fair")

    assert len(initial) == 5
    assert len(pool) == 371
    assert 353 in set(pool["data_index"])
    assert set(initial["data_index"]) == {33, 162, 164, 244, 288}


def test_sensitivity_pool_excludes_complete_bo_trajectory():
    candidates, trajectory = load_inputs()

    pool, _ = build_candidate_pool(candidates, trajectory, pool_definition="exclude_bo")

    assert len(pool) == 323
    assert set(pool["data_index"]).isdisjoint(set(trajectory["data_index"]))


def test_seed_42_reproduces_archived_exclude_bo_lists():
    candidates, trajectory = load_inputs()

    result = run_baseline_seed(
        candidates,
        trajectory,
        seed=42,
        pool_definition="exclude_bo",
    )

    random_selected = result.loc[
        (result["method"] == "random") & (result["stage"] > 0), "data_index"
    ].tolist()
    stratified_selected = result.loc[
        (result["method"] == "stratified_random") & (result["stage"] > 0),
        "data_index",
    ].tolist()

    assert random_selected[:8] == [144, 335, 53, 123, 257, 21, 252, 216]
    assert stratified_selected[:8] == [5, 362, 333, 21, 149, 54, 145, 69]
    assert len(random_selected) == len(set(random_selected)) == 48
    assert len(stratified_selected) == len(set(stratified_selected)) == 48


def test_selection_is_yield_blind():
    candidates, trajectory = load_inputs()
    changed = candidates.copy()
    changed["yield_percent"] = np.arange(len(changed), dtype=float)

    original = run_baseline_seed(candidates, trajectory, seed=7, pool_definition="fair")
    modified = run_baseline_seed(changed, trajectory, seed=7, pool_definition="fair")

    cols = ["method", "stage", "selection_order", "data_index"]
    pd.testing.assert_frame_equal(original[cols], modified[cols])


def test_each_method_uses_shared_five_plus_two_rounds_of_24():
    candidates, trajectory = load_inputs()

    result = run_baseline_seed(candidates, trajectory, seed=0, pool_definition="fair")

    counts = result.groupby(["method", "stage"]).size().to_dict()
    assert counts == {
        ("random", 0): 5,
        ("random", 1): 24,
        ("random", 2): 24,
        ("stratified_random", 0): 5,
        ("stratified_random", 1): 24,
        ("stratified_random", 2): 24,
    }
