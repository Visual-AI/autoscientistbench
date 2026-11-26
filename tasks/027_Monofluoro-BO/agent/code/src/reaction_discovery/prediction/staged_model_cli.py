#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Official staged model primitives; benchmark data and save/predict entry points are in main.py."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


MODEL_ORDER = ["xgboost", "rf", "svm"]
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "rf": "RandomForest",
    "svm": "SVM",
}
PKL_ALIASES = {
    "xgboost": {
        "classifier": ["xgboost_classifier.pkl"],
        "regressor": ["xgboost_regressor.pkl"],
        "scaler": [],
    },
    "rf": {
        "classifier": ["rf_classifier.pkl", "randomforest_classifier.pkl"],
        "regressor": ["rf_regressor.pkl", "randomforest_regressor.pkl"],
        "scaler": [],
    },
    "svm": {
        "classifier": ["svm_classifier.pkl"],
        "regressor": ["svm_regressor.pkl"],
        "scaler": ["svm_scaler.pkl"],
    },
}




def parse_model_list(raw: str) -> List[str]:
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in models if item not in MODEL_ORDER]
    if invalid:
        raise ValueError(f"Unsupported model families: {', '.join(invalid)}")
    ordered = [name for name in MODEL_ORDER if name in models]
    if not ordered:
        raise ValueError("No valid model families were provided.")
    return ordered


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip()
        lower = key.lower()

        if lower in {"sub_smiles1", "sub1", "substrate1", "sub1_smiles"} or (
            "sub" in lower and "1" in lower and "smiles" in lower
        ):
            rename_map[col] = "sub_smiles1"
        elif lower in {"sub_smiles2", "sub2", "substrate2", "sub2_smiles"} or (
            "sub" in lower and "2" in lower and "smiles" in lower
        ):
            rename_map[col] = "sub_smiles2"
        elif lower in {"prod_smiles", "prod", "product", "product_smiles"} or (
            "prod" in lower or "product" in lower
        ):
            rename_map[col] = "prod_smiles"
        elif lower in {"yield", "yield%", "actual_yield"} or "yield" in lower:
            rename_map[col] = "yield"
        elif lower in {"split"}:
            rename_map[col] = "split"
        elif lower in {"cls_label", "class_label", "label"}:
            rename_map[col] = "cls_label"
        elif key in {"底物1", "底物1smiles"}:
            rename_map[col] = "sub_smiles1"
        elif key in {"底物2", "底物2smiles"}:
            rename_map[col] = "sub_smiles2"
        elif key in {"产物", "产物smiles"}:
            rename_map[col] = "prod_smiles"
        elif key in {"产率", "产率%", "实际产率"}:
            rename_map[col] = "yield"

    return df.rename(columns=rename_map)


def ensure_required_columns(df: pd.DataFrame, require_yield: bool) -> None:
    needed = {"sub_smiles1", "sub_smiles2", "prod_smiles"}
    if require_yield:
        needed.add("yield")
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")


def load_csv(path: Path, require_yield: bool) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df = normalize_columns(df)
            ensure_required_columns(df, require_yield=require_yield)
            return df
        except Exception as exc:  # pragma: no cover - best effort loading
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def morgan_fp(smiles: str, radius: int, n_bits: int) -> Optional[np.ndarray]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for SMILES featurization. Install it before training or prediction."
        ) from exc

    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def featurize_dataframe(
    df: pd.DataFrame, bits: int, radius: int, require_yield: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    kept_rows = []
    features = []
    targets = []

    if require_yield:
        df = df.copy()
        df["yield"] = to_float_series(df["yield"])
        df = df.dropna(subset=["yield"])

    for _, row in df.iterrows():
        fp1 = morgan_fp(row["sub_smiles1"], radius=radius, n_bits=bits)
        fp2 = morgan_fp(row["sub_smiles2"], radius=radius, n_bits=bits)
        fp3 = morgan_fp(row["prod_smiles"], radius=radius, n_bits=bits)
        if fp1 is None or fp2 is None or fp3 is None:
            continue
        features.append(np.concatenate([fp1, fp2, fp3]).astype(np.float32))
        kept_rows.append(row)
        if require_yield:
            targets.append(float(row["yield"]))

    if not features:
        raise RuntimeError("No valid rows remained after SMILES featurization.")

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    X = np.vstack(features)
    y = np.array(targets, dtype=np.float32) if require_yield else None
    return X, y, kept_df




def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
        }
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def get_xgboost_classes():
    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for xgboost models. Install it with `pip install xgboost`."
        ) from exc
    return XGBClassifier, XGBRegressor


def build_model_bundle(model_name: str, random_state: int) -> Dict[str, object]:
    if model_name == "rf":
        return {
            "classifier": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state,
                n_jobs=int(os.environ.get("BENCH_CPUS", "8")),
            ),
            "regressor": RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=random_state,
                n_jobs=int(os.environ.get("BENCH_CPUS", "8")),
            ),
            "scaler": None,
        }

    if model_name == "svm":
        return {
            "classifier": SVC(kernel="rbf", C=1.0, gamma="scale"),
            "regressor": SVR(kernel="rbf", C=1.0, gamma="scale"),
            "scaler": StandardScaler(),
        }

    if model_name == "xgboost":
        XGBClassifier, XGBRegressor = get_xgboost_classes()
        return {
            "classifier": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=random_state,
                n_jobs=int(os.environ.get("BENCH_CPUS", "8")),
                verbosity=0,
            ),
            "regressor": XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=random_state,
                n_jobs=int(os.environ.get("BENCH_CPUS", "8")),
                verbosity=0,
                objective="reg:squarederror",
            ),
            "scaler": None,
        }

    raise ValueError(f"Unsupported model family: {model_name}")


def transform_features(
    scaler: Optional[StandardScaler], X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if scaler is None:
        return X_train, X_test
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def run_staged_prediction(
    classifier: object,
    regressor: object,
    scaler: Optional[StandardScaler],
    X_train: Optional[np.ndarray],
    X_test: np.ndarray,
    y_train: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if scaler is not None:
        if X_train is None:
            X_pred = scaler.transform(X_test)
        else:
            X_train_scaled, X_pred = transform_features(scaler, X_train, X_test)
            classifier.fit(X_train_scaled, (y_train > 0).astype(int))
            regressor.fit(X_train_scaled[y_train > 0], y_train[y_train > 0])
            pred_cls = classifier.predict(X_pred)
            pred_yield = np.zeros(len(X_test), dtype=np.float32)
            nonzero_mask = pred_cls.astype(bool)
            if nonzero_mask.any():
                pred_yield[nonzero_mask] = regressor.predict(X_pred[nonzero_mask])
            pred_yield = np.clip(pred_yield, 0.0, None)
            return pred_cls.astype(int), pred_yield
        pred_cls = classifier.predict(X_pred)
        pred_yield = np.zeros(len(X_test), dtype=np.float32)
        nonzero_mask = pred_cls.astype(bool)
        if nonzero_mask.any():
            pred_yield[nonzero_mask] = regressor.predict(X_pred[nonzero_mask])
        pred_yield = np.clip(pred_yield, 0.0, None)
        return pred_cls.astype(int), pred_yield

    if X_train is not None and y_train is not None:
        classifier.fit(X_train, (y_train > 0).astype(int))
        regressor.fit(X_train[y_train > 0], y_train[y_train > 0])

    pred_cls = classifier.predict(X_test)
    pred_yield = np.zeros(len(X_test), dtype=np.float32)
    nonzero_mask = pred_cls.astype(bool)
    if nonzero_mask.any():
        pred_yield[nonzero_mask] = regressor.predict(X_test[nonzero_mask])
    pred_yield = np.clip(pred_yield, 0.0, None)
    return pred_cls.astype(int), pred_yield




def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def save_model_family(output_dir: Path, model_name: str, result: Dict[str, object]) -> None:
    alias = PKL_ALIASES[model_name]
    save_pickle(result["classifier"], output_dir / alias["classifier"][0])
    save_pickle(result["regressor"], output_dir / alias["regressor"][0])
    if alias["scaler"] and result["scaler"] is not None:
        save_pickle(result["scaler"], output_dir / alias["scaler"][0])


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def find_existing_file(model_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        path = model_dir / name
        if path.exists():
            return path
    return None


def load_model_family(model_dir: Path, model_name: str) -> Dict[str, object]:
    alias = PKL_ALIASES[model_name]
    cls_path = find_existing_file(model_dir, alias["classifier"])
    reg_path = find_existing_file(model_dir, alias["regressor"])
    scaler_path = find_existing_file(model_dir, alias["scaler"])

    if cls_path is None or reg_path is None:
        raise FileNotFoundError(
            f"Could not find required PKL files for {model_name} in {model_dir}."
        )

    return {
        "classifier": load_pickle(cls_path),
        "regressor": load_pickle(reg_path),
        "scaler": load_pickle(scaler_path) if scaler_path is not None else None,
    }




def write_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)








