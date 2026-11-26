# WEPP surveillance lineage abundance estimation

## 1. Goal
Estimate RSV-A lineage proportions from the supplied wastewater-surveillance short reads using the fixed reference genome and mutation-annotated tree. Improve agreement with known mixture proportions, measured by constrained lineage-abundance RMSE.

## 2. Data
Archived paired-read observations are in data/val/input, with public labels in data/val/labels.json. The mixture-scale partition is fixed. Every supplied sample retains its original read pairs. Read identifiers and sample filenames are opaque, and paired observations are shuffled together. Sequence and quality bytes are preserved. data/train/reference contains the fixed reference and tree, but there is no training split for fitting reusable models or learned preprocessing. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Per-sample inference and deconvolution may use that sample's observations and the fixed reference/tree. Generating haplotype sequences are not supplied.

## 3. Metric and scoring
RMSE is the square root of mean squared abundance error over the union of expected names and estimated names with abundance at least 0.005, with missing estimates zero. Average equally over sample instances. Each sample must miss at most one expected lineage at the0.5% threshold and assign at most2% abundance mass to non-simulated lineages. Frozen scorer enforces both guards; failure gives +inf. No extra abundance renormalization is applied. Validator rejects invalid values/duplicates/sample omissions. Avoiding outside data and output hardcoding requires patch review.

Measured validation RMSE is 0.0022178753325867886, with standard deviation 0.0 from one reference run.

## 4. Baseline
The baseline rebuilds WEPP and its native clustering helper from code/. It places reads on the supplied tree, selects candidate haplotypes, and uses Freyja with CLARABEL to fit nonnegative proportions to the sample's depth-weighted allele frequencies. It prunes low-abundance candidates, adds nearby haplotypes, and iteratively refines the estimates. Haplotype proportions are aggregated into lineage abundances. The original thresholds are baseline settings. Optional diagnostic BAM export and sequence generation are absent. Recorded compilation took 14.44 seconds, and validation prediction took 547.95 seconds.

## 5. Interface
bash code/train.sh --out DIR builds artifacts; bash code/predict.sh --model DIR --input PATH --output FILE analyzes supplied observations. See bench/io_spec.md. BENCH_DATA_DIR locates fixed reference inputs and BENCH_CPUS controls bounded jobs/threads. The pipeline is deterministic conditional on observations and numerical libraries; BENCH_SEED is supplied by the harness. bench eval-dev uses the same frozen validator/scorer. Snapshot dependencies are copied readonly and env.lock is immutable.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Editable surface is code/. Data,bench/,README and I/O are frozen. Candidate source must actually execute; precompiled original WEPP and external editable installs are unavailable. No network or original reproduction paths are mounted. The allocation is up to 16 CPUs and 64 GB. Output lineage-abundance estimates only. The same scientific guards apply to full inputs; smoke mode does not silently replace those inputs.

## 7. Known pitfalls
There are no new simulated reads or reconstructed haplotypes in this task. Header scrubbing and pair shuffling prevent source identities/order from exposing labels. Freyja imports resolve to candidate source, and its relative native-helper path requires the supplied directory layout. Small read subsets may miss expected lineages and legitimately receive +inf under unchanged guards.

## Research scope

**Objective.** Estimate lineage abundances from the existing surveillance observations using the fixed reference/tree and minimize the original constrained abundance RMSE.

**Fixed research objects.** existing paired-read observations, reference/tree, lineage-name universe and mixture-scale partition; union-of-lineages RMSE without extra normalization, missing-lineage and spurious-mass guards.

**Editable methods.** computational abundance estimators, read summarization and statistical inference procedures; optimization and implementation with the same observed inputs.

**Forbidden substitutions.** When retaining the supplied WEPP pipeline, changing only MIN_PROP=0.005, MIN_AF=0.005, MIN_DEPTH=10, MIN_Q=20, MIN_LEN=80, MAX_NEIGHBOR_ITERATIONS=10, TOP_N=10, MAX_PEAKS=300, MAX_NEIGHBORS_WEPP=50 or MAX_PEAK_PEAK_MUTATION=2 does not count as a method improvement. Also forbidden: changing observed sequences, reference lineages, expected-composition definitions or sample grouping; generating new biological observations, using hidden compositions or replacing estimates with archived answers. Report only lineages the estimator infers as present. Padding the output with zero or negligible abundances to enlarge the RMSE denominator does not count as a method improvement.

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
