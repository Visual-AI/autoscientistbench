# PRSBridge benchmark I/O

`--input` is an uncompressed NumPy NPZ (allow_pickle=False). `id` is an array
of N unique opaque strings in the required output order. `chr1` through `chr22`
are uint8 arrays with shape (V_c, ceil(N/4)). Each byte contains four two-bit
PLINK codes in increasing bit order: 00 means two copies of allele a1,
01 is missing, 10 means one copy, 11 means zero copies. Any unused tail bits
must be ignored. SNP order and alleles are supplied by
`data/train/variantsC.npz`: arrays `snp`, `a1`, `a2`, all length V_c.
The baseline maps missing dosages to zero.
An individual's genotype is the only prediction input for that individual;
opaque identifiers carry no predictive information.

The output is CSV **text**, even if its path has no extension, with exactly
`id,pred_0,pred_1` as the header. Provide one row per supplied id, in exactly
the input order, with two finite real-valued PRS candidates. No extra rows,
columns, missing rows or duplicates are allowed. A single candidate may be
repeated in both columns. The two candidate functions are fixed across all
scoring splits. They need not retain the baseline's hyperparameter meanings.

Labels are NPZ with `id` (N,), `y` (N,) float64 phenotype and `tune` (100,N)
boolean masks. Each mask marks the tuning half; its complement is the scoring
half. These masks are read only by the frozen scorer. Validation labels are public only for evaluation and method, hyperparameter or
checkpoint selection. Do not fit models or learned preprocessing on validation
features or labels. Prediction inputs contain neither phenotypes nor masks.

For each of the 100 splits, the scorer computes squared Pearson correlations
between each candidate and phenotype on each half, selects the candidate
with the largest tuning correlation squared (ties choose the first), and
records its correlation squared on the other half. `mean_r2` is the mean of
these 100 values. This is correlation R², not one minus normalized squared
prediction error. Invalid files or undefined/nonfinite correlations return
`Infinity` and fail the submission. No quality threshold beyond this metric
is imposed.

Training summaries live at `data/train/sumdat/chrC/hm3_sumdat.txt` with columns
SNP, A1, A2, BETA, SE, N. The LD reference at `data/train/ref/chrC` consists
of `block.txt`, per-block BIM metadata, and NumPy correlation matrices.
Training writes a model directory; the baseline saves per-chromosome candidate
coefficient tables, build/fit logs and `model.json`. All required model artifacts
must live under `--out` so prediction works in a fresh process.

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
