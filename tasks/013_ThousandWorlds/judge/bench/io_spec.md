# Frozen tensor I/O

All arrays use NumPy NPZ with `allow_pickle=False`. Each key starts with one of
`multi-partial__`, `multi-complete__`, `single-complete__`.

Input contains `id` (sorted original simulation row IDs), `X` (N by 8 raw planetary
parameters), `s` (integer simulator category), and `field_names` (F public names).
Features, in order: T_star (K), F_star (W/m²), radius (m), gravity (m/s²),
P_rot (Earth days), P0 (Pa), CO2 and CH4 (volume mixing ratios).

Output contains exactly `id` and `pred` for each subset. IDs must match input order
exactly, with no duplicates. `pred` is a finite floating array of shape
(5, N, F, 32, 64): five independent fitted predictors, in algorithm seed order;
these are not five draws to average into one ensemble. Predictions are raw physical
grids (K, kg/kg humidity, unitless cloud fraction, m/s winds, W/m² radiative fluxes).
F=53 for multi-partial and F=48 for complete subsets. Labels hold the truth grid,
whole-field observation mask, public field names and eight frozen Train-mean RMSEs.
Missing truth fields contain NaN and are excluded; predictions must remain finite.
Write NPZ through an open binary file so arbitrary output names work.

Training: `data/train/SUBSET/train.npz` includes id, X, simulator index s, planet_id,
Y_grid, Y_spectral (raw preprocessed spherical harmonics), field_mask (spectral),
grid_mask, field_names, raw_field_names, input_names, n_sim_types and sh_mask.
`stats.pkl` is the authors' Stats object fitted on the training groups only; it is
trusted provided data, not a prediction format. The baseline reads only this split.

Scoring: compute float32 latitude-weighted RMSE per example and field, average
examples with each field observed, average vertical levels within each of eight
variables, divide by the corresponding Train-mean RMSE, then take the geometric
mean. Finally average these scalar scores across the five independent fits.
Score is multi-partial if multi-complete ≤ 0.48 and single-complete ≤ 0.55;
otherwise exactly 1.0. Malformed/non-finite outputs return Infinity and fail validation.
The normalization is fixed in labels, never estimated from submitted predictions.

## Validation use (baseline and candidates)

Fit models and learned preprocessing only on the designated training observations.
Validation features and labels are for evaluation and method, hyperparameter or
checkpoint selection only. This restriction applies equally to the baseline and
every candidate, including final retraining. Do not merge validation into training,
fit normalization/encoders on validation, distill from validation predictions, or
relabel validation observations as training or calibration data.

For tasks defined as per-instance inference, optimization or online interaction,
use only the observation/training/calibration/feedback roles explicitly specified
for that instance. Do not use its evaluation targets or carry fitted state from
validation cases into other independent cases. A development replay directory may
contain separately designated training observations; its validation observations
remain excluded from fitting. Public visibility does not authorize training use.
