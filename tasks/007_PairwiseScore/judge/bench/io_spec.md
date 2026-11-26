# Ranked comparisons to density samples

`train.sh --out DIR` stores method configuration. `predict.sh --model DIR --input PATH --output FILE` fits each instance independently and writes full samples. Inputs and outputs are NPZ (no pickle); output must use exactly FILE, even without a filename extension.

Input keys: `ids` is a unique opaque string vector. For each ID `i`, `i__dataset` is float32 `(2,D,N)` in winner/loser order, already transformed exactly as the released experiment prescribes. `i__bounds`, `i__means`, `i__covariance_matrix` describe the known proposal transform (empty means/covariances for uniform proposals). `i__sigma_data` is the released empirical model preconditioning scale: for untransformed 2D data it comes from the original preliminary 10,000-point calibration sample; otherwise it is the transformed comparison standard deviation. No calibration sample rows or evaluation target samples are inputs.

Prediction keys: `ids` plus one float32 or float64 matrix named exactly for each ID. Each matrix has D columns and 25,000 rows at D=2/4, 30,000 at D=6, 40,000 at D=10/16. Coordinates are in original units. All entries must be finite. Instance order is arbitrary; each array's row order is the sampler order because Wasserstein uses its leading rows. No scalar metrics are accepted as predictions. A cloud collapsed to one point cannot define the required KDE and is rejected by the scorer. Exact copies of evaluation-target marginal samples, including row permutations, violate the observations-only protocol and are also rejected; broader truth-table misuse requires patch audit.

Labels: `ids`, and `i__target` (evaluation samples), `i__group`, `i__seed`, `i__w_count`, and frozen `i__baseline_w`. Public training labels have no baseline field. The scorer requires frozen baselines, so it does not score them; metric computation itself can be used independently for diagnostics. Group identities/seeds/targets are evaluation information, never permissible prediction inputs. Do not reconstruct generator seeds or use identity lookups.

Each instance's predictions must depend only on its own comparisons, supplied proposal/calibration information, a fixed configuration selected on the public instances and BENCH_SEED. Other instances, labels, target generators, and memorized density tables cannot be used. See README for patch-audit limitations.

Research scope: The fixed research object is density inference from these ranked comparisons and their known proposal/calibration metadata. Density estimators, score networks, training surrogate losses and samplers may change; target generators, observation provenance, original-coordinate sample counts and the frozen MMTV/Wasserstein criteria may not. Use proposal means, proposal covariances and `sigma_data` only as the baseline does, to map between original and unit-cube coordinates and to precondition the score model; do not use them to set the location, scale or component positions of the estimate (patch audit). Apply one estimator family to every instance; numeric hyperparameters and the released preprocessing may depend on dimension, but no choice may depend on which of the eight targets an instance appears to be (patch audit). The sampler keeps the fitted input-dependent tempering field tau(x): do not replace it by a constant such as its mean or quantile, the paper's score-tau* variant, on any instance, including by forcing tau(x) to be constant; a gain must come from a better tau(x), score model or sampler (patch audit).

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
