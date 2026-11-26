# 062_SkewPosterior input/output protocol

Input is UTF-8 JSON with `version: 1` and a nonempty `cases` array. Each case contains an opaque `id`, observed `X` (n by d array), observed count vector `y`, `n`, `d`, prior degrees of freedom `nu: 3`, and `fit_seed` for reproducible fitting. The five original (n,d) pairs are (200,60), (400,84), (600,102), (800,120), (1000,132). The reference is never included in inputs. There is no cross-instance training dataset. Fit each case only from its own observed X/y. Validation is for evaluation and method or hyperparameter selection, without fitting reusable models or learned preprocessing.

Output is UTF-8 JSON with `version: 1` and exactly one entry per input case, in any order. Each entry has exactly `id`, `mean` (d finite posterior-mean estimates), `full_fit_evals` (nonnegative integer), `wall_s` (positive finite seconds), and `reference_accesses` (nonnegative integer). IDs must match exactly, without duplicates. JSON duplicate keys and NaN/infinity are invalid.

`full_fit_evals` includes every log-posterior invocation during optimizer initialization, line search, Hessian, and skewing. The baseline instruments both compiled Stan log_prob_impl overloads and the R skew callback. `wall_s` records elapsed inference time per fit as a diagnostic (optimizer-through-skew-weight time in the baseline). The baseline uses 10,000 Gaussian draws and 20,000 skew log-posterior calls, plus compiled calls; alternative inference procedures are allowed for the same target posterior.

Labels contain the archived HMC coordinate mean/SD and the same-case archived plain Laplace median bias and S-LAP wall seconds. The scorer computes the median coordinate absolute standardized error per case, averages over replicates within each n, then averages over n. If any fit exceeds 21,500 evaluations or reports any reference access, the entire score becomes the same archive's plain Laplace aggregate.

The scorer checks reported numbers. Their truthful provenance and zero reference use require patch audit. The hidden labels are isolated by the sandbox. Candidate code must never open any reference labels, including public validation labels, during fitting. No hard-coded or cached predictions.

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
