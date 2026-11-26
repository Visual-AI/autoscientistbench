# CLEAR interval I/O

The input is an NPZ archive loaded with `allow_pickle=False`. For each dataset `d`, `x_d` is a float64 matrix of feature rows and `id_d` is a same-length vector of opaque strings. Features retain the source preprocessing and column order. There are ten algorithm repetitions, indexed 0 through 9; all use the supplied fixed partition. Row order and input length may vary.

Output is UTF-8 CSV with exactly `id,pred,lower,upper`. For every input id and repetition r, emit one row whose id is the input id followed by `_r` and r. `pred` is the central estimate, the baseline's PCS median; `lower` and `upper` are the final calibrated interval bounds in target units. All numbers must be finite, and lower <= pred <= upper. Every expected id must occur exactly once; row ordering is arbitrary. Write exactly to the output path, which need not have an extension.

`data/train/train.npz` contains `x_d` and `y_d` per dataset. Validation inputs are in `data/val/input.npz`; `data/val/labels.npz` contains matching `id_d` and `y_d`.

Scoring uses the released NCIW calculation at target coverage .95: compute the smallest empirical scaling factor covering .95 of targets about `pred`, multiply by mean original width, and divide by the range of targets (unless the range is zero). Compute this separately for every dataset and repetition. Average repetitions within datasets, then average all 17 datasets equally. The PICP guard uses the submitted bounds before scoring rescaling: the ten-repetition mean coverage must be at least .87 on test and .90 on val, and at most .99, for every dataset. A row whose target lies beyond a zero-width side of its interval has an infinite scaling factor. A violation returns `Infinity`. Diagnostic keys report unguarded NCIW, coverage, and per-dataset results. Scoring rescaling never exposes targets to prediction code.

## Research scope

Produce narrow calibrated regression intervals at fixed 95% target coverage on the 17 supplied datasets.

Fixed research objects and target: The datasets, partitions, regression targets/units and ten-repetition evaluation remain fixed. The NCIW formula/rescaling and per-dataset PICP at least 0.87 on test and 0.90 on val and at most 0.99 remain fixed.

Editable methods: The interval estimator, calibration method, internal features, training surrogate and optimizer of the uncertainty method may change using allowed training/calibration inputs; the regression model behind `pred` may not.

Forbidden substitutions: Do not change coverage targets, data, splits, target units, width formula or coverage guard; do not calibrate on hidden labels. The central estimate `pred` must remain the baseline's PCS median: the model pool, selection and bootstrap ensemble that produce it stay fixed, and only the uncertainty estimators and their calibration may change (patch audit).

The baseline reserves a deterministic 20% subset within each dataset's supplied
training rows for model selection and interval calibration (RandomState seed
20261102). The remaining training rows fit estimators. Neither the validation
features nor validation targets enter those fits or calibration.

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
