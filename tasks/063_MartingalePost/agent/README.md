# 063_MartingalePost: parametric martingale posterior

## 1. Task
Compute central 95% credible intervals for mu1,s1,s12 under the specified bivariate-normal parametric martingale posterior. Baseline uses the original hybrid predictive resampling with B=2000,T=50, sampler seed 382, and exact sorted-draw indices 50/1950.

## 2. Data
Validation has 5000 datasets for each n=20,100,500 in data/val/input, with labels in data/val/labels.json. Inputs are observed 2xn matrices, with shuffled opaque IDs and binary storage offsets. Validation uses original seed 1533+i and retains shared-seed pairing across n. The fixed design parameters are public; the estimator must use observations rather than truth. No training split is supplied. Validation is for evaluation and selection only, except that per-instance inference uses that instance's observed matrix.

## 3. Metric
For each interval, IS=hi-lo+40max(lo-theta,0)+40max(theta-hi,0). Average over datasets, scale by sqrt(n), average the nine (n,parameter) cells. Lower is better. A pass requires relative score improvement of at least max(0.005,2*reference_std/abs(reference_mean)). Outputs must cover every input ID with finite ordered endpoints and finite nonnegative elapsed_s. Prediction time is diagnostic, with no score threshold. Truth nonaccess and timing provenance require patch audit. There are no separate coverage checks. The measured validation reference has mean 5.611644397499667 and standard deviation 0.0 over one run. Training took 0.17 s and validation prediction took 111.42 s.

## 4. Compute
Julia 1.10.5 runtime and pinned Distributions dependency closure are copied under the task runtime. Unused plotting/benchmark imports are removed; numerical function bodies are unchanged. Dependencies are fixed; no online installation. Small smoke uses one original-size dataset per n with full sampler settings.

## 5. Commands
bash code/train.sh --out DIR writes a marker. From code/, run bash predict.sh --model DIR --input PATH --output FILE to construct candidate intervals. Use absolute paths for DIR, PATH and FILE.

## 6. Changes
The bivariate-normal predictive family and its natural-gradient martingale posterior remain fixed, initialized by each dataset's sample mean and unbiased covariance. The update learning rule 1/N defines the posterior and cannot be tuned into a different posterior. Inference, approximation and numerical algorithms may change while computing the same central 95% posterior intervals. Hybrid resampling, B=2000 and T=50 are baseline numerical choices; quantile levels remain 2.5% and 97.5%. Use each dataset's observed data only and retain the fixed interval score. Posterior provenance requires source audit. Runtime snapshots/env.lock are immutable.

## 7. Source
Upstream eadc71bac56148b100de8c63f37d68f718b37e78, src/normal_biv.jl plus current metric driver. The adapter moves generation offline, reads observations and writes the selected intervals. Original hybrid functions and RNG seed remain unchanged.

## Research contract

**Objective:** Approximate the specified bivariate-normal parametric martingale posterior and its central 95% intervals, evaluated by the unchanged scaled interval score.

**Fixed research objects:** The predictive family N(mu,Sigma), sample-mean/unbiased-covariance initialization, natural-gradient update Z and learning rule 1/N define the target limiting posterior. The mu1/s1/s12 parameters, 2.5%/97.5% quantiles, independent observed datasets, fixed n grid and sqrt(n)-scaled interval score remain fixed. Write theta=(mu1,mu2,s1,s2,s12), where s1 and s2 are variances and s12 is covariance. For each N>n, draw Y from N(mu,Sigma) at the previous state, set d=Y-mu, and update theta by (d1,d2,d1^2-s1,d2^2-s2,d1*d2-s12)/N. All entries use the previous state. The target posterior is the distribution of the limiting theta conditional on the observed dataset.

**Editable methods:** Predictive-resampling implementations, truncation/tail approximations, direct integration or other numerical inference algorithms may change while approximating that posterior. Sample count and numerical solver settings may change as part of a new numerical method; an auxiliary loss can improve approximation but cannot redefine the predictive law or posterior.

**Forbidden substitutions:** Do not replace this posterior with a conventional Bayesian prior, generic frequentist interval or changed predictive model/learning rule to obtain easier coverage. No true design parameters as fitted values, cross-dataset pooling or modification of the interval score. When retaining the supplied hybrid predictive resampling, changing only B=2000, T=50 or sampler seed 382 does not count as a method improvement.

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
