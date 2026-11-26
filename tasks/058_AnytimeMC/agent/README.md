# 058_AnytimeMC: Anytime-valid Monte Carlo multiple testing

`train.sh --out MODEL` prepares the candidate method; the supplied baseline is training-free. `predict.sh --model MODEL --input DESCRIPTOR --output ARTIFACT` exports code and configuration as a ZIP policy artifact. The trusted scorer runs `policy.sh` in a fresh worker for each independent trial, using the immutable R 4.4.3 snapshot. The supplied baseline uses `sequential_BH`. Candidate sampling/test strategies, auxiliary models/losses, stopping rules and multiple-testing procedures under `code/` may be replaced for the same sampler-only hypotheses, provided they use the same sampler-only observations and meet the frozen FDR, power and query-cost evaluation.

The outer input is a `private-null-values-v3` JSON descriptor. A worker receives one CSV with an anonymous trial ID, hypothesis index, the **original observed statistic**, B and alpha. `bench/sampler.R` provides the original raw Gaussian null values through `open_sampler(trial_id,M,B)`, `sampler[indices,1:t]` and `sampler$counts()`. The original column-major random stream and floating-point values are preserved. Large prefix requests are split into transport batches without changing their values or sampling cost. Null RNG states, unexposed samples, latent effects and truth remain private to the scorer. Open one sampler per trial; creating another object does not reset the service or its counts.

Use unique in-range hypothesis indices and cumulative prefixes within B. Re-reading an exposed prefix adds no draws. The service retains the longest disclosed prefix for each hypothesis and verifies the candidate's `draws`. Worker output CSV columns are `trial_id,hypothesis,rejected,draws,sampler_only`; set sampler_only=1 only for compliant methods. Emit exactly one row per hypothesis. `rejected` is binary, and `draws` is a positive integer no larger than B. Unsupported operations, budget violations and false counts invalidate the submission. Empty index selections retain the original empty-array behavior and disclose no samples.

The original full experiment is ten trials with M=1000, B=10000 and alpha=0.1. The objective averages unique disclosed permutations per hypothesis, then across trials; lower is better. Power is the mean across trials of true alternative rejections divided by the number of true alternatives. FDR is the mean across trials of null rejections divided by total rejections. Both denominators are floored at one. Mean power must be at least 0.720 and mean FDR at most 0.10; the penalty is 10000. The original h=10 Besag-Clifford baseline, endpoint BH rule and unused b=0.9 parameter are unchanged.

Only the original observed statistics and authorized null samples may enter evaluated decisions. Do not use analytic Gaussian p-values, simulate extra null draws outside the sampler, reconstruct truth, reuse validation labels as answers, or retain information across independent trials. These are information-access restrictions of the Monte Carlo sampling problem, not a requirement to use the baseline algorithm; they require source/patch review. The service enforces private-state isolation and disclosure counts; a numerical score or sampler_only declaration does not certify mathematical method compliance. The scorer explicitly reports `patch_audit_required` and `method_compliance_verified=false`.

No training split is provided. Public observations, validation labels and validation null states in `data/val/evaluation` support evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable model state or learned preprocessing on validation observations or labels. Per-trial computation may use only that trial's observed statistics and authorized sampler responses. Runtime, env.lock, data, scorer and sampler interfaces are frozen. BENCH_SEED controls candidate randomness independently of the fixed sampler streams. The resource ceiling is 32 CPUs and 16 GB.

The measured validation mean cost is 134.7097 permutations per hypothesis, with standard deviation 0.0 across one reference run. That run had power 0.7281 and FDR 0.0665, so it passed both guards. Training took 0.11 s and validation artifact export took 0.17 s. Validation scoring time was not recorded separately.

## Research contract

**Objective:** Reduce the actual Monte Carlo null-sampling cost of the specified multiple-testing problem while maintaining its fixed power and FDR thresholds. Each hypothesis tests a zero mean against a positive shift.

**Fixed research objects:** Only supplied observed statistics and authorized raw null-sampler prefixes are admissible decision information; the null law cannot be replaced by analytic Gaussian knowledge or external draws. Ten independent trials, M=1000, B=10000, alpha=.1, actual unique disclosed query cost, power>=.720 and FDR<=.10 remain fixed.

**Editable methods:** Adaptive query allocation, stopping, valid p-value/e-value construction, multiple-testing strategy and auxiliary fitting methods may change within the sampler-only information contract. Besag-Clifford h=10 and sequential-BH implementation are baseline methods rather than mandatory algorithms.

**Forbidden substitutions:** No analytical normal p-values, extra simulated null samples, sampler-RNG recovery, inaccessible future prefixes, false query counts or cross-trial state. When retaining the supplied sequential_BH code, changing only method="bc" (including its existing bm and AMT branches), param_bc=10 or peeking_times=1:B does not count as a method improvement. Do not replace null hypotheses, change FDR/power definitions or submit known truth labels as decisions. Rejections must be supported by Monte Carlo p-values, e-values or another valid test calibrated from authorized null-sampler observations. A rule that rejects by an observed-statistic cutoff or rank chosen through validation selection, while ignoring the disclosed null samples, is not allowed. Patch audit enforces this.

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
