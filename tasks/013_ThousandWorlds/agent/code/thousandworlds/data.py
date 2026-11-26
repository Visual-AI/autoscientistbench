from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray
    field_mask_train: np.ndarray
    field_mask_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    field_names: list[str]
    input_names: list[str]
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    space: str
    subset: str
    protocol: str
    raw_field_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.raw_field_names = list(self.field_names) if self.raw_field_names is None else list(self.raw_field_names)
        self.field_names = list(self.field_names)

