# 057_BT-SBM observed-data interface
Input JSON is exactly `{"datasets":[{"W":[[0,1],[2,0]]}, ...]}`. Each W is an integer nonnegative directed win-count matrix, square with zero diagonal. The full task uses 105 players per dataset. W + transpose(W) is the observed exposure. Inputs contain no truth partition, true strength, K, design label, replication identifier, seed, or player name. Dataset and player orders must be preserved in output.

Output JSON is exactly `{"datasets":[{"partition":[1,2],"strength":[0.5,2.0],"fit_time_s":1.2}, ...]}`. One output per input, in order. Partition is the minimum posterior expected-VI point estimate (consecutive integer labels from 1 through the number of predicted groups); strength is the normalized posterior player-strength summary under the specified BT-SBM. Any valid inference algorithm for that fixed posterior and those summaries may be used. Strength values must be finite and positive; only their ranking affects the metric. Fit time is positive elapsed inference seconds, excluding compilation, reported as a diagnostic. Duplicate keys, NaN/Infinity, wrong counts, missing or extra keys are rejected.

The scorer recomputes ARI and tie-aware Spearman from private labels, then returns mean ARI only if mean Spearman >=0.8211555622531486; otherwise 0. The ranking threshold is fixed. The scorer checks reported vector/timing validity; observed-data provenance and truthful timing require patch audit.

Exact metric compatibility: the frozen implementation preserves the original installed fossil::adj.rand.index, including its absolute value and unequal-cluster-count behavior. It does not substitute conventional signed ARI.

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
