"""Frozen verbatim upstream Table 4 metric functions (float32 reductions)."""
import numpy as np
import re
from pathlib import Path
SPLIT_VAR_REGEX = re.compile(r"^(.+?)_(\d+)$")
VARIABLE_ORDER = ('surface_temperature','temperature','specific_humidity_dex','cloud_fraction','u','v','asr','olr')
def load_latitude_weights():
    return np.load(Path(__file__).with_name('latitude_weights.npy'), allow_pickle=False)
def public_name(name):
    parts=name.rsplit('_',1)
    base=parts[0] if len(parts)==2 and parts[1].isdigit() else name
    suffix=f'_{parts[1]}' if base!=name else ''
    return {'asr_cloudy':'asr','olr_cloudy':'olr'}.get(base,base)+suffix

def _lat_weights() -> np.ndarray:
    w = load_latitude_weights().astype(np.float32)
    return w / w.sum()


def _sanitize_grid(arr: np.ndarray, field_mask: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if field_mask is None:
        return arr
    mask = np.asarray(field_mask, dtype=bool)[None, ...] if arr.ndim == 5 else np.asarray(field_mask, dtype=bool)
    while mask.ndim < arr.ndim:
        mask = mask[..., None]
    return np.where(mask, np.nan_to_num(arr, nan=0.0), 0.0)


def _per_field_mean_with_mask(per_example_per_field: np.ndarray, field_mask: np.ndarray | None) -> np.ndarray:
    if field_mask is None:
        return per_example_per_field.mean(axis=0)
    mask = np.asarray(field_mask, dtype=np.float32)
    count = mask.sum(axis=0)
    total = (per_example_per_field * mask).sum(axis=0)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _base_var(name: str) -> str:
    match = SPLIT_VAR_REGEX.match(name)
    return match.group(1) if match else name


def _variable_key(name: str) -> str:
    base = _base_var(name)
    if base == "specific_humidity":
        return "specific_humidity_dex"
    return public_name(base)


def _field_metric_name(name: str, *, dex_specific_humidity: bool) -> str:
    name = public_name(name)
    return f"{name}_dex" if dex_specific_humidity and name.startswith("specific_humidity") else name


def _variable_means(per_field: np.ndarray, field_names: list[str]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, name in enumerate(field_names):
        value = float(per_field[i])
        if value != value:
            continue
        key = _variable_key(name)
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in VARIABLE_ORDER if counts.get(key, 0) > 0}


def _metric_space(arr: np.ndarray, field_names: list[str], *, humidity: str | None) -> np.ndarray:
    """Convert humidity fields before scoring when metrics are reported in log space."""
    out = np.asarray(arr, dtype=np.float32).copy()
    if humidity is None:
        return out
    for i, name in enumerate(field_names):
        if not name.startswith("specific_humidity"):
            continue
        if humidity == "ln":
            out[..., i, :, :] = np.log(np.clip(out[..., i, :, :], 1.0e-12, None))
        elif humidity == "dex":
            out[..., i, :, :] = np.log10(np.clip(out[..., i, :, :], 1.0e-12, None))
        else:
            raise ValueError(f"Unknown humidity metric space: {humidity}")
    return out


def _weighted_mse_per_example_field(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.square(pred - target)
    lon_mean = error.mean(axis=-1)
    return (lon_mean * _lat_weights()[None, None, :]).sum(axis=-1)


def _result(per_field: np.ndarray, field_names: list[str] | None, *, dex_specific_humidity: bool = False) -> dict:
    per_variable = _variable_means(per_field, field_names) if field_names is not None else {}
    return {
        "per_field": (
            dict(zip([_field_metric_name(name, dex_specific_humidity=dex_specific_humidity) for name in field_names], per_field.tolist()))
            if field_names is not None
            else per_field
        ),
        "per_variable": per_variable,
    }


def rmse(
    pred: np.ndarray,
    target: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
    *,
    dex_specific_humidity: bool = False,
) -> dict:
    pred = _sanitize_grid(pred, field_mask)
    target = _sanitize_grid(target, field_mask)
    per_field = _per_field_mean_with_mask(np.sqrt(_weighted_mse_per_example_field(pred, target) + 1.0e-12), field_mask)
    return _result(per_field, field_names, dex_specific_humidity=dex_specific_humidity)
