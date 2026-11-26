# 061_DP-FRT: DP-FRT-Bayes credible sets

## 1. Task
Estimate the 95% credible set of the nonprivate Fisher randomization p-value from privatized binary counts. The baseline is the authors' DP-FRT-Bayes sampler with uniform prior, 10000 posterior draws, alpha=0.05 and clipped noisy counts.

## 2. Data
Validation contains 36,000 independent releases: four effect cases, three total sample sizes (100/500/1000) and epsilon 0.2/0.5/1, with 1000 replications per cell. Inputs contain the treated/control group sizes n1,n0, privacy budget epsilon, clipped noisy success counts t11,t01 and opaque shuffled IDs. Validation inputs and labels are in data/val/; bench/generator.R defines the public validation design. The public val generator seed is 1022026. There is no training set. Validation is for evaluation and selection only, not fitting reusable state. Per-instance posterior inference uses only that release's supplied observations.

## 3. Metric and guards
Minimize mean interval width averaged over the 36 cells. The scorer returns 1.0 unless every cell has coverage >=0.91 and mean cell coverage >=0.94. Coverage means the true p_FRT belongs to the discrete credible set after rounding to 12 decimals. Scorer-checkable guards include completeness, finite [0,1] intervals and discrete membership. The scorer reports mean_coverage for the higher-coverage tie break; the harness decision uses the scalar primary metric. Across replay seeds, let r, s and c be the reference mean, reference standard deviation and candidate mean. Passing requires (r-c)/abs(r) >= max(0.005, 2*s/abs(r)). The measured validation mean width is 0.3877489214, with standard deviation 0.0000214890.

## 4. Compute
The installed R 4.4.3 and matrixStats 1.5.0 runtime is copied into an immutable snapshot. Thread count follows BENCH_CPUS. The baseline performs posterior computation during predict. Small smoke runs all 36 cells with one observation each and retains 10000 draws. For validation inference seeds 1/2/3 on 12 CPUs, training took 0.11/0.07/0.06 seconds and validation prediction took 272.98/267.37/268.09 seconds.

## 5. Commands
`bash code/train.sh --out DIR` writes a marker for this training-free method. From `code/`, run `bash predict.sh --model DIR --input PATH --output FILE` to compute the candidate uncertainty set. Use absolute paths for DIR, PATH and FILE.

## 6. Allowed changes
The DP-FRT-Bayes posterior under the discrete uniform count prior and supplied geometric privacy mechanism is fixed. Sampling, exact enumeration or other numerical posterior approximations may replace the baseline sampler using only the available privatized observations. The target remains the original nonprivate Fisher randomization p-value. Hidden labels, raw private counts and latent privacy noise are inaccessible; do not substitute known case-specific answers. Respect the input/output contract and coverage constraints. Dependencies are fixed by the copied runtime, so env.lock must stay unchanged.

## 7. Source
Original repository qy-sun/dp_frt, upstream commit c8af4ea430d6a44e70884267482e6987a9fe32b0. DP-FRT.R is copied from the existing reproduction. The protocol-only change accepts already-privatized observations, moving noise generation to the frozen offline generator; the posterior body is unchanged. The supplied baseline retains its prior, draw count and internal clipping. The uniform prior, mechanism likelihood, central 95% posterior-set definition and scorer's 12-decimal membership rule remain fixed; numerical inference choices may change. Source review must verify these semantics beyond the output-format checks.

## Research contract

**Objective:** Compute the central 95% credible set of the nonprivate Fisher randomization p-value under the specified DP-FRT-Bayes denoising posterior, minimizing its guarded width.

**Fixed research objects:** The discrete uniform prior over valid treated/control count pairs and two-sided geometric privacy likelihood with supplied epsilon define the fixed posterior. The Fisher hypergeometric right-tail p-value, central 2.5%/97.5% posterior quantiles and supported discrete set between them, 12-decimal membership, 36-cell width/coverage rules and privatized inputs remain fixed. For integer counts a in 0:n1 and b in 0:n0, normalize weights exp(-epsilon*(abs(t11-a)+abs(t01-b))). Map each pair to phyper(a-1, a+b, n1+n0-a-b, n1, lower.tail=FALSE). Aggregate equal p-values after rounding to 12 decimals before computing the central quantiles and their supported set.

**Editable methods:** Posterior sampling, exact enumeration, integration, caching of input-independent calculations and other numerical approximation may change. Numerical routines may improve; auxiliary losses must approximate the same posterior rather than create a different uncertainty set.

**Forbidden substitutions:** Do not change the count prior, privacy likelihood or p-value tail; replace credible sets with arbitrary frequentist intervals/HPD sets; or choose case-specific answers. No original counts, latent noise, hidden case labels or additional privacy releases may be used. When retaining the supplied DP-FRT-Bayes sampler, changing only R=10000 or method='sampling' does not count as a method improvement.

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
