"""Yield-blind retrospective baselines for the 376-condition campaign."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INITIAL_BUDGET = 5
ROUND_BUDGET = 24
STAGE_BUDGETS = {0: 5, 1: 29, 2: 53}
STRATIFY_COLUMNS = ("cat_family", "base_class", "solvent_class", "temp_class")
YIELD_THRESHOLDS = (0, 10, 30, 50)


def normalize_text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def infer_cat_family(catalyst) -> str:
    value = normalize_text(catalyst)
    if not value:
        return "unknown"
    for family in ("Pd", "Ni", "Cu", "Ag", "Rh", "Ir", "Ru", "Co", "Fe", "Au", "Pt"):
        if re.search(rf"\b{family}\b", value) or value.startswith(family):
            return family
    match = re.match(r"([A-Za-z]+)", value)
    return match.group(1) if match else value


def infer_base_class(base) -> str:
    value = normalize_text(base).replace("·", ".").replace(" ", "")
    if not value:
        return "unknown"
    if "CO3" in value or "HCO3" in value:
        return "carbonate/bicarbonate"
    if "PO4" in value or "H2PO4" in value or "HPO4" in value:
        return "phosphate"
    if value in {"KOH", "NaOH", "CsOH", "LiOH"} or value.endswith("OH"):
        return "hydroxide"
    if "OAc" in value or "AcO" in value:
        return "acetate"
    if value in {"LiCl", "KCl", "NaCl", "LiBr", "KBr", "NaBr"}:
        return "halide"
    if "OtBu" in value or "OMe" in value or "OEt" in value:
        return "alkoxide"
    if value in {"NMM", "DIPEA", "Et3N", "NEt3", "DBU", "DABCO"}:
        return "amine"
    if "SO4" in value:
        return "sulfate"
    return "other"


def infer_solvent_class(solvent) -> str:
    value = normalize_text(solvent)
    groups = {
        "aromatic": {"toluene", "benzene", "xylene", "mesitylene"},
        "ether": {"1,4-dioxane", "THF", "2-MeTHF", "DME", "MTBE"},
        "halogenated": {"DCE", "DCM", "CHCl3", "chlorobenzene"},
        "nitrile": {"MeCN", "acetonitrile"},
        "polar_aprotic": {"DMF", "DMA", "NMP", "DMSO"},
        "alcohol": {"MeOH", "EtOH", "iPrOH", "tBuOH", "HFIP"},
        "water": {"H2O", "water"},
        "alkane": {"n-hexane", "hexane", "heptane", "cyclohexane"},
    }
    for class_name, members in groups.items():
        if value in members:
            return class_name
    return value or "unknown"


def add_class_columns(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    temperature_column = "Temp./C" if "Temp./C" in result.columns else "Temp./℃"
    required = {"Cat.", "base", "Sol.", temperature_column}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    result["Cat."] = result["Cat."].map(normalize_text)
    result["base"] = result["base"].map(normalize_text)
    result["Sol."] = result["Sol."].map(normalize_text)
    result["cat_family"] = result["Cat."].map(infer_cat_family)
    result["base_class"] = result["base"].map(infer_base_class)
    result["solvent_class"] = result["Sol."].map(infer_solvent_class)
    result["temp_class"] = pd.to_numeric(result[temperature_column], errors="coerce").map(
        lambda x: "unknown" if pd.isna(x) else f"{x:g}"
    )
    return result


def as_percent(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values.astype(str).str.replace("%", "", regex=False), errors="coerce")
    if numeric.notna().any() and numeric.max() <= 1.5:
        numeric = numeric * 100
    return numeric.fillna(0.0)


def normalize_candidates(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "Temp./C" not in result.columns and "Temp./℃" in result.columns:
        result = result.rename(columns={"Temp./℃": "Temp./C"})
    if "data_index" not in result.columns:
        result.insert(0, "data_index", np.arange(len(result), dtype=int))
    if "yield_percent" not in result.columns:
        yield_column = next((name for name in ("Yield", "yield") if name in result.columns), None)
        if yield_column is None:
            raise ValueError("Candidate table must contain yield_percent, Yield, or yield.")
        result["yield_percent"] = as_percent(result[yield_column])
    else:
        result["yield_percent"] = as_percent(result["yield_percent"])
    result["data_index"] = pd.to_numeric(result["data_index"], errors="raise").astype(int)
    if result["data_index"].duplicated().any():
        raise ValueError("Candidate data_index values must be unique.")
    return add_class_columns(result)


def normalize_trajectory(table: pd.DataFrame) -> pd.DataFrame:
    required = {"round", "data_index"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"BO trajectory is missing columns: {sorted(missing)}")
    result = table.copy()
    result["round"] = pd.to_numeric(result["round"], errors="raise").astype(int)
    result["data_index"] = pd.to_numeric(result["data_index"], errors="raise").astype(int)
    if result["data_index"].duplicated().any():
        raise ValueError("BO trajectory data_index values must be unique.")
    return result


def build_candidate_pool(
    candidates: pd.DataFrame,
    trajectory: pd.DataFrame,
    pool_definition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = normalize_candidates(candidates)
    trajectory = normalize_trajectory(trajectory)
    initial_indices = trajectory.loc[trajectory["round"] == 0, "data_index"].tolist()
    if len(initial_indices) != INITIAL_BUDGET:
        raise ValueError(f"Expected {INITIAL_BUDGET} shared initialization conditions.")
    indexed = candidates.set_index("data_index", drop=False)
    missing = set(trajectory["data_index"]).difference(indexed.index)
    if missing:
        raise ValueError(f"Trajectory indices missing from candidate table: {sorted(missing)}")
    initial = indexed.loc[initial_indices].reset_index(drop=True)
    if pool_definition == "fair":
        excluded = set(initial_indices)
    elif pool_definition == "exclude_bo":
        excluded = set(trajectory["data_index"])
    else:
        raise ValueError("pool_definition must be 'fair' or 'exclude_bo'.")
    pool = candidates.loc[~candidates["data_index"].isin(excluded)].copy()
    return pool, initial


def _compute_targets(pool: pd.DataFrame, n: int, columns: Iterable[str]) -> dict:
    targets = {}
    for column in columns:
        proportions = pool[column].value_counts(normalize=True, dropna=False)
        raw = proportions * n
        target = np.floor(raw).astype(int)
        missing = n - int(target.sum())
        if missing:
            for value in (raw - target).sort_values(ascending=False).index[:missing]:
                target.loc[value] += 1
        targets[column] = target.to_dict()
    return targets


def _stratified_sample(
    pool: pd.DataFrame,
    n: int,
    seed: int,
    columns: Iterable[str] = STRATIFY_COLUMNS,
    top_k: int = 8,
) -> pd.DataFrame:
    columns = tuple(columns)
    rng = np.random.default_rng(seed)
    available = pool.copy()
    targets = _compute_targets(pool, n, columns)
    counts = {column: {} for column in columns}
    selected = []
    for _ in range(n):
        scored = []
        for index, row in available.iterrows():
            score = 0.0
            for column in columns:
                value = row[column]
                target = targets[column].get(value, 0)
                if target > 0:
                    score += (target - counts[column].get(value, 0)) / target
            scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        candidates = [index for index, _ in scored[: min(top_k, len(scored))]]
        chosen = int(rng.choice(candidates))
        row = available.loc[chosen]
        selected.append(chosen)
        for column in columns:
            value = row[column]
            counts[column][value] = counts[column].get(value, 0) + 1
        available = available.drop(index=chosen)
    return pool.loc[selected].copy()


def _label_selection(table: pd.DataFrame, method: str, stage: int, start: int) -> pd.DataFrame:
    result = table.copy()
    result.insert(0, "selection_order", np.arange(start, start + len(result), dtype=int))
    result.insert(0, "stage", stage)
    result.insert(0, "method", method)
    return result


def run_baseline_seed(
    candidates: pd.DataFrame,
    trajectory: pd.DataFrame,
    seed: int,
    pool_definition: str = "fair",
    top_k: int = 8,
) -> pd.DataFrame:
    pool, initial = build_candidate_pool(candidates, trajectory, pool_definition)
    if len(pool) < 2 * ROUND_BUDGET:
        raise ValueError("Candidate pool is smaller than the 48-condition baseline budget.")

    rng = np.random.default_rng(seed)
    random_indices = rng.choice(pool.index.to_numpy(), size=2 * ROUND_BUDGET, replace=False)
    random_rows = pool.loc[random_indices]

    stratified_round1 = _stratified_sample(pool, ROUND_BUDGET, seed + 1001, top_k=top_k)
    remaining = pool.loc[~pool["data_index"].isin(stratified_round1["data_index"])]
    stratified_round2 = _stratified_sample(remaining, ROUND_BUDGET, seed + 2002, top_k=top_k)

    blocks = []
    for method in ("random", "stratified_random"):
        blocks.append(_label_selection(initial, method, 0, 1))
    blocks.extend(
        [
            _label_selection(random_rows.iloc[:ROUND_BUDGET], "random", 1, 6),
            _label_selection(random_rows.iloc[ROUND_BUDGET:], "random", 2, 30),
            _label_selection(stratified_round1, "stratified_random", 1, 6),
            _label_selection(stratified_round2, "stratified_random", 2, 30),
        ]
    )
    result = pd.concat(blocks, ignore_index=True)
    result.insert(0, "seed", int(seed))
    result.insert(0, "pool_definition", pool_definition)
    return result.sort_values(["method", "selection_order"], kind="stable").reset_index(drop=True)


def run_many_seeds(
    candidates: pd.DataFrame,
    trajectory: pd.DataFrame,
    seeds: Iterable[int] = range(50),
    pool_definitions: Iterable[str] = ("fair", "exclude_bo"),
) -> pd.DataFrame:
    runs = [
        run_baseline_seed(candidates, trajectory, seed, pool_definition)
        for pool_definition in pool_definitions
        for seed in seeds
    ]
    return pd.concat(runs, ignore_index=True)


def summarize_runs(selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = []
    for (pool_definition, seed, method), group in selections.groupby(
        ["pool_definition", "seed", "method"], sort=True
    ):
        for stage, budget in STAGE_BUDGETS.items():
            subset = group.loc[group["stage"] <= stage]
            yields = subset["yield_percent"]
            row = {
                "pool_definition": pool_definition,
                "seed": int(seed),
                "method": method,
                "budget": budget,
                "best_yield": float(yields.max()),
            }
            for threshold in YIELD_THRESHOLDS:
                row[f"count_gt_{threshold}"] = int((yields > threshold).sum())
            per_seed.append(row)
    per_seed = pd.DataFrame(per_seed)

    aggregate_rows = []
    metric_columns = ["best_yield", *(f"count_gt_{x}" for x in YIELD_THRESHOLDS)]
    for keys, group in per_seed.groupby(["pool_definition", "method", "budget"], sort=True):
        for metric in metric_columns:
            values = group[metric]
            aggregate_rows.append(
                {
                    "pool_definition": keys[0],
                    "method": keys[1],
                    "budget": int(keys[2]),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "q025": float(values.quantile(0.025)),
                    "q975": float(values.quantile(0.975)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "n_seeds": int(values.size),
                }
            )
    return per_seed, pd.DataFrame(aggregate_rows)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)
