# Diffusivity design interface

The submission is a spatial diffusivity profile, not a density or a claimed metric.

## Input

`--input PATH` is an NPZ file, loaded with `allow_pickle=False`:

| Key | Shape | Meaning |
|---|---|---|
| id | (n,) strings | grid-point identifiers |
| positions | (n,) float64 | periodic coordinates in Å |
| reduced_potential | (n,) float64 | fixed ABF PMF divided by kBT |
| system | scalar string | physical system identifying its public ABF statistics |

Training data have one directory per physical system: observed segment lengths/counts and sorted cell keys; `statistics.npz` contains the interval/position grid, counts, censoring fractions and blocking SEM metadata. `meta.json` records sampling period, sampling walls, box length and temperature. `context.npz` and `pmf_abf.txt` carry public physical inputs. No fitted diffusivity or NLL is supplied.

## Output

UTF-8 CSV, header exactly `id,diffusivity`. Exactly one row for each input id, no duplicates or extras; row order is immaterial. Diffusivity is finite and strictly positive, in Å²/ps. Use 17 significant digits. No extra fields, PMF replacement, densities, or claimed metrics are accepted. The output path need not have an extension.

## Labels and score

`labels.npz` contains the physical inputs plus `lags` and `density` (lag × position). Public validation labels describe POPC. Reference densities for evaluation are held by the judge.

`python bench/validate.py PRED INPUT` checks the format. `python bench/scorer.py PRED LABELS` propagates the submitted design using the fixed ABF PMF, periodic boundaries, and initial position at the central cell. The frozen modules are verbatim copies of the official `pdda` solver and density comparator.

`tv` is the mean total variation at 75 and 200 ps if mean TV across all supplied lags is at most 0.0435 on test or 0.18 on val, otherwise 1. Invalid designs produce Infinity. The fixed evaluator enforces the PMF/input rule by construction. `tv_raw`, `tv_all`, `quality_guard_ok`, and `tv_by_lag` are diagnostics. POPC has its own available lag list. Its validation score saturates only when its all-lag mean exceeds the validation cap stated above.

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
