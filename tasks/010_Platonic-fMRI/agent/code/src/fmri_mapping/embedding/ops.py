import pandas as pd
import numpy as np
import torch
from typing import Optional, Union
from pathlib import Path
import gc
import os

from ..io.nsd import get_resource, get_subject_roi, get_subject_behavioural


def int16_scaled_quantiles(raw, probabilities, scaling=300.0):
    """Exact NumPy linear quantiles after monotonic float32 conversion/scaling."""
    histogram = np.zeros(65536, dtype=np.int64)
    flat = raw.reshape(-1)
    for start in range(0, raw.size, 1 << 20):
        histogram += np.bincount(
            flat[start:start+(1 << 20)].astype(np.int32) + 32768,
            minlength=65536)
    cumulative = np.cumsum(histogram)
    ranks = np.asarray(probabilities) * (raw.size - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    lo = np.searchsorted(cumulative, lower, side="right") - 32768
    hi = np.searchsorted(cumulative, upper, side="right") - 32768
    pairs = np.stack([lo, hi], axis=1).astype(np.float32)
    if scaling is not None:
        pairs /= scaling
    return np.array([np.quantile(pair, fraction)
                     for pair, fraction in zip(pairs, ranks-lower)])


def get_subject_betas(
    subject: int,
    roi: int,
    q: float = 0.004,
    center: bool = True,
    normalize: bool = False,
    betas_subfolder: str = "betas",
    low_memory: bool = True,
    return_quantiles: bool = False,
    base_dir: Optional[str | Path] = None,
    scaling: float | None = 300.0,
    qcut: Optional[Union[float, list[float]]] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Get the betas for a subject and roi, with optional normalization.
    Use the NSD volumetric dataset
    """

    betas = get_subject_roi(
        subject=subject, roi=roi, base_dir=base_dir, subfolder=betas_subfolder
    )
    # Raw int16 order statistics can be computed exactly without partitioning
    # the much larger float array. Scaling and NumPy interpolation stay intact.
    histogram_quantiles = None
    if (os.environ.get("REPRO_INT16_QUANTILES") == "1" and
            betas.dtype == np.int16 and dtype == np.float32 and
            qcut is None and q > 0 and not low_memory and
            (scaling is None or scaling > 0)):
        histogram_quantiles = int16_scaled_quantiles(betas, [q, 1-q], scaling)
    # dtype
    betas = betas.astype(dtype)
    # collect garbage
    gc.collect()

    if scaling is not None:  # Reduce by 300 following the NSD documentation
        betas /= scaling

    if qcut is not None:  # Use a fixed value instead of quantiles
        quantiles = [-qcut, qcut] if isinstance(qcut, float) else qcut
        assert len(quantiles) == 2, "qcut should be a float or a list of two floats"
        betas = np.clip(betas, qcut[0], qcut[1], dtype=dtype)
    elif q > 0 and low_memory:
        # Compute quantiles for each row and then global quantiles
        row_quantiles = np.quantile(betas, [q, 1 - q], axis=1)
        q0 = np.quantile(row_quantiles[0].ravel(), 0.5)
        q1 = np.quantile(row_quantiles[1].ravel(), 0.5)
        quantiles = np.array([q0, q1])
        betas = np.clip(betas, quantiles[0], quantiles[1], dtype=dtype)
    elif q > 0:
        quantiles = (histogram_quantiles if histogram_quantiles is not None else
                     np.quantile(betas.ravel(), [q, 1 - q], axis=0))
        betas = np.clip(betas, quantiles[0], quantiles[1], dtype=dtype)

    if center:  # Center the betas
        betas -= np.mean(betas, axis=0)

    if normalize:  # Optionally make norm 1 each row
        betas /= np.clip(np.linalg.norm(betas, axis=1, keepdims=True), 1e-6, None)

    if return_quantiles:
        return betas, quantiles

    return betas


def get_confound_matrix(
    subject: int, confound_column: str = "session_run"
) -> np.ndarray:
    d_runs = get_subject_behavioural(subject)
    C = pd.get_dummies(d_runs[confound_column], drop_first=False).to_numpy(
        dtype=np.float32
    )
    return C


def split_repetitions(
    subject: int,
    shuffle_indexes: bool = True,
    random_state=None,
    how="outer",
    min_exists: int = 2,
) -> pd.DataFrame:
    if random_state is None:
        random_state = int(os.environ.get("BENCH_SEED", "42"))
    df_stimuli = get_resource("stimulus")

    df_stimuli = df_stimuli.query(f"subject == {subject}")
    if how == "inner":
        df_stimuli = df_stimuli.query("exists")
        df_stimuli = df_stimuli[["nsd_id", "shared", "subject_index", "repetition"]]
    else:
        df_stimuli = df_stimuli.copy()
        df_stimuli.loc[~df_stimuli.exists, "subject_index"] = -1
        df_stimuli = df_stimuli[["nsd_id", "shared", "subject_index", "repetition"]]
        # Nullable int type
        df_stimuli.subject_index = df_stimuli.subject_index.astype("Int64")
        df_stimuli.subject_index = df_stimuli.subject_index.replace(-1, np.nan)

    if shuffle_indexes:
        df_stimuli = df_stimuli.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
        df_stimuli["repetition"] = df_stimuli.groupby("nsd_id").cumcount()
        df_stimuli = df_stimuli.sort_values("subject_index").reset_index(drop=True)

    df_stimuli_1 = (
        df_stimuli.query("repetition == 0")
        .rename(columns={"subject_index": "subject_index_1"})
        .drop(columns=["repetition"])
    )
    df_stimuli_2 = (
        df_stimuli.query("repetition == 1")
        .rename(columns={"subject_index": "subject_index_2"})
        .drop(columns=["repetition", "shared"])
    )
    df_stimuli_3 = (
        df_stimuli.query("repetition == 2")
        .rename(columns={"subject_index": "subject_index_3"})
        .drop(columns=["repetition", "shared"])
    )
    df_stimuli_merged = df_stimuli_1.merge(df_stimuli_2, on="nsd_id", how=how).merge(
        df_stimuli_3, on="nsd_id", how=how
    )

    df_stimuli_merged["exists"] = (
        df_stimuli_merged[["subject_index_1", "subject_index_2", "subject_index_3"]]
        .notna()
        .sum(axis=1)
    )

    df_stimuli_merged = df_stimuli_merged.query(f"exists >= {min_exists}").reset_index(
        drop=True
    )

    return df_stimuli_merged.copy()


def split_betas_by_subject_index(betas, df_repetitions, inpute=False):

    if not inpute:
        df_repetitions = df_repetitions.query("exists == 3").reset_index(drop=True)
        subject_index_1 = df_repetitions["subject_index_1"].astype(int).to_numpy()
        subject_index_2 = df_repetitions["subject_index_2"].astype(int).to_numpy()
        subject_index_3 = df_repetitions["subject_index_3"].astype(int).to_numpy()
        betas_c = betas

    if inpute == "zeros" or inpute == "mean":
        nan_vector = np.full(betas.shape[1], fill_value=np.nan, dtype=betas.dtype)
        betas_c = np.vstack([betas, nan_vector])
        last = betas_c.shape[0] - 1
        subject_index_1 = (
            df_repetitions["subject_index_1"].fillna(last).astype(int).to_numpy()
        )
        subject_index_2 = (
            df_repetitions["subject_index_2"].fillna(last).astype(int).to_numpy()
        )
        subject_index_3 = (
            df_repetitions["subject_index_3"].fillna(last).astype(int).to_numpy()
        )

    # Split into 3 (zeros if inpute)
    betas_1 = betas_c[subject_index_1, :]
    betas_2 = betas_c[subject_index_2, :]
    betas_3 = betas_c[subject_index_3, :]

    if inpute == "mean":
        nan_mask = df_repetitions[
            ["subject_index_1", "subject_index_2", "subject_index_3"]
        ]
        nan_mask = nan_mask.isna().to_numpy()
        assert nan_mask.sum(axis=1).max() <= 1, "Only one missing repetition allowed"

        mask = nan_mask[:, 0]
        if mask.any():
            betas_1[mask, :] = (betas_2[mask, :] + betas_3[mask, :]) / 2

        mask = nan_mask[:, 1]
        if mask.any():
            betas_2[mask, :] = (betas_1[mask, :] + betas_3[mask, :]) / 2

        mask = nan_mask[:, 2]
        if mask.any():
            betas_3[mask, :] = (betas_1[mask, :] + betas_2[mask, :]) / 2

    return betas_1, betas_2, betas_3


def to_tensor(*arrays, device="cuda", dtype=torch.float):
    return [torch.tensor(a, device=device, dtype=dtype) for a in arrays]
