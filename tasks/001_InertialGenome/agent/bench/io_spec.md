# Input/output contract

Inputs are NPZ files loaded with `allow_pickle=False`. `ids` is a one-dimensional array of opaque structure identifiers, in evaluation order. For its zero-based position i, `coords_i` is a float64 array of shape (N,3) with supplied LorDG coordinates, and `nodes_i` is an int64 array of shape (N,) with local locus indices. The ids do not encode genomic chromosome identities. Each structure is independent. Preprocessing belongs to the method: the baseline multiplies coordinates by 100, applies its inertial frame, divides by maximum vector norm, and pads to 2400.

Training and validation each have `input.npz` and `labels.npz`. Labels have matching `ids`, `contact_i` (N,N) contact matrices, and scalar `n_i`. Contact matrices are supervision, never prediction inputs. Inference must use only the supplied structure's LorDG coordinates, node indices and the trained model. It may not use validation labels to recognize evaluation inputs.

Prediction is a UTF-8 CSV with exact header `id,slot,x,y,z`. Emit structures in input id order, then slots 0 through 2399 for each. Each id/slot pair occurs exactly once. All coordinates must be finite float32-representable numbers. CSV is written at the exact output path, irrespective of extension. Output coordinates are the model's internal coordinates, before multiplying by input scale. All 2400 slots are required, including padding: the authors normalize distances by the maximum over the entire padded matrix. This behavior is intentionally retained.

The scorer computes float32 `torch.cdist` for all slots and divides by the full matrix maximum. For contacting, non-diagonal real-locus pairs (contact greater than 1e-8), targets are contact^(-0.5), divided by their maximum within the structure. It computes SciPy Spearman correlation and float32 RMSE over both directed halves, then averages equally across structures. Mean dRMSE must not exceed 0.095 on test and 0.12 on val. Undefined metrics, malformed predictions, or failed guards return `dscc: Infinity` (the benchmark rejection signal); diagnostic `scientific_score` records the worst value -1.

`validate.py PRED INPUT` rejects wrong headers, missing/extra/reordered/duplicate ids or slots, wrong column counts, and nonfinite values. `scorer.py PRED LABELS` prints one JSON line with primary metric `dscc` and diagnostics. Input provenance and absence of label access require patch audit; finite predictions alone cannot prove them.

## Research scope

Refine the supplied LorDG chromosome structures to improve contact-derived geometric agreement.

Fixed research objects and target: The supplied LorDG coordinates, node indices, splits and contact-derived target distances are fixed. The equal-structure dSCC objective, float32 padded 2400-slot distance normalization and dRMSE at most 0.095 on test and 0.12 on val remain fixed.

Editable methods: The refinement model, internal coordinate features, training surrogate, optimizer and hyperparameters may change; inference uses only the current structure and trained model.

Forbidden substitutions: Do not substitute another initial structure source, use contacts as prediction inputs, or change normalization, padding, metric or guard. Real-locus coordinates must come from a trained refinement model; returning the supplied coordinates unchanged, or changed only by a rigid motion or a global scale, is not a valid submission.

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
