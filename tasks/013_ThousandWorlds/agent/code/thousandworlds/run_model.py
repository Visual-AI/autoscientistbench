from __future__ import annotations
import numpy as np
import thousandworlds as tw

def _gplfr_inverse_preprocess_grid(Y: np.ndarray, field_names: list[str], stats: tw.Stats, field_idxs: list[int]) -> np.ndarray:
    out = np.asarray(Y, dtype=np.float32).copy()
    for j in field_idxs:
        strategy_name, kwargs = tw.preprocessing._strategy_params(field_names[j], stats)
        out[:, j] = tw.preprocessing._inverse_preprocess_array(out[:, j], strategy_name, kwargs)
    return out


def _gplfr_decode_public_predictions(coeffs: np.ndarray, data) -> np.ndarray:
    field_names = data.spectral_bundle.raw_field_names
    coeffs_norm = np.transpose(coeffs, (0, 1, 3, 2)).astype(np.float32, copy=False)
    scales = np.ones(coeffs_norm.shape[1:3], dtype=np.float32)
    if data.stats.asr_olr_normalize_by_f_star:
        f_star = data.grid_bundle.X_test[:, data.stats.input_names.index("F_star")].astype(np.float32)
        for j, name in enumerate(field_names):
            if tw.preprocessing._base_var(name) in tw.preprocessing.ASR_OLR_FIELDS:
                scales[:, j] = f_star
    cloud_idxs = [i for i, name in enumerate(field_names) if name.startswith("cloud_fraction")]
    export_idxs = [i for i in range(len(field_names)) if i not in cloud_idxs]
    predictions = []
    for sample in coeffs_norm:
        sample_coeffs = tw.unnormalise_spectral(sample, field_names, data.stats) * scales[:, :, None]
        sample_coeffs = tw.apply_symmetry_mask(sample_coeffs, field_names, data.sh_mask.T)
        grid = tw.to_grid(sample_coeffs, data.inverse_sht).astype(np.float32)
        grid = _gplfr_inverse_preprocess_grid(grid, field_names, data.stats, cloud_idxs)
        if cloud_idxs:
            grid[:, cloud_idxs] = np.clip(grid[:, cloud_idxs], 0.0, 1.0)
        predictions.append(_gplfr_inverse_preprocess_grid(grid, field_names, data.stats, export_idxs))
    out = np.stack(predictions, axis=0)
    if cloud_idxs:
        out[:, :, cloud_idxs] = np.clip(out[:, :, cloud_idxs], 0.0, 1.0)
    return out.astype(np.float32)

