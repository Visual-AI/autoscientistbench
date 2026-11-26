# 042_FrontDoor-NP: Front-door causal-effect estimation

## Goal

Improve estimation of the average treatment effect (ATE) of binary treatment A on outcome Y from observed X, A, M and Y. M is a binary mediator that carries the treatment effect. An unobserved variable may confound A and Y. Treatment overlap is weak, so some treatment probabilities approach the extremes. Lower mean squared error is better, subject to nominal 95% confidence interval coverage of at least 0.887.

The front-door mean for treatment level t is
`psi(t) = E_X[sum_m P(M=m | A=t,X) sum_a E(Y | M=m,A=a,X) P(A=a | X)]`.
The ATE is the treated front-door mean minus the untreated front-door mean. Both sums run over the binary levels. This observed-data functional, rather than ordinary treatment-outcome association, is the target.

## Paper and baseline

Guo, Benkeser and Nabi, *Flexible nonparametric inference for causal effects under
the front-door model*. The baseline is the paper's front-door TMLE, binary mediator,
n=500, logistic-regression mediator model, propensity truncation 0.001/0.999. In the baseline, cross-fitting
and SuperLearner are disabled and the targeting iteration cap is 15000. Candidate
nuisance working models, surrogate fitting losses, cross-fitting and estimation algorithms may differ while preserving the same front-door ATE and observed-data identification problem.

The supplied R runtime includes `fdcausal`. All files in `code/` can be changed; the frozen scorer and input data define the task independently of the baseline implementation.

## Data and score

Each CSV case contains 500 observed records (X,A,M,Y and derived AX,MX); the latent
confounder and ground-truth ATE are excluded. The validation split contains 1000
datasets generated with public seeds 1–1000. No population truth or labels may be used in the predictor.

The scorer returns mean squared ATE error if coverage is >=0.887; otherwise it
returns +Infinity. Coverage uses strict interval bounds, matching the authors'
aggregation. This guard is checked by the scorer. Restriction to observed inputs
and prohibition on hardcoded answers are checked through code review/isolation.

Measured validation MSE is 0.006044116097063796, with standard deviation 0.0 across three reference runs. Validation coverage is 0.912. Marker creation took 0.06 seconds in each run. Validation prediction took 11.79, 11.58 and 11.34 seconds, respectively.

## Resources

32 CPU cores, 128 GiB RAM, default research budget 5 hours. `BENCH_CPUS` controls
parallel prediction. The task uses a frozen copied R environment and needs no network.

## Interfaces

`bash code/train.sh --out DIR` creates a marker without fitting a shared model. There is no shared training split. Estimation occurs during
`bash code/predict.sh --model DIR --input FILE --output FILE`.

Fit each case and its nuisance functions only from that case's observed records. Validation is only for evaluation and method or hyperparameter selection. Do not pool cases to fit reusable models or learned preprocessing, or carry fitted state between independent cases.

Output is CSV `case_id,estimate,lower,upper`, with exactly one finite result per
input case. See `bench/io_spec.md`. `RSCRIPT` is provided by the snapshot
runtime. `BENCH_SEED`, `BENCH_CPUS`, and `BENCH_DATA_DIR` follow the benchmark contract.

## Rules

Use the provided observed inputs and CPU budget. Do not read hidden evaluation files, hardcode
ground truth, modify the frozen scorer/validator, or use earlier reproduction files. The method may be replaced within `code/`.

## Submission

Use the benchmark CLI's `eval-dev` and `submit` workflow. Submit code changes. The supplied runtime, data and evaluator are fixed.

## Research contract

**Objective:** Improve front-door ATE estimation under the fixed weak-overlap binary-mediator experiment while maintaining nominal 95% interval coverage.

**Fixed research objects:** The front-door causal identification problem and ATE, binary mediator, n=500 observed X/A/M/Y and weak-overlap generator remain fixed. Mean squared ATE error, strict interval inclusion, coverage>=.887, supplied datasets and independent-case inference remain fixed.

**Editable methods:** Nuisance models, surrogate losses, density/propensity estimation, cross-fitting, targeting or other valid ATE estimators and numerical methods may change. TMLE versus one-step/other valid front-door procedures are methods for this fixed estimand. Truncation and computational settings may be retuned as part of such a method change.

**Forbidden substitutions:** Do not switch to ordinary confounded regression association, direct effect or another mediation estimand; add latent confounders/true ATE as inputs; or change outcomes/splits. No oracle constants, hidden data, extra labeled simulations or changed metric/coverage. When retaining the supplied estfd pipeline, changing only truncate_lower=0.001, truncate_upper=0.999, n.iter=15000 or cvg.criteria=0.001 does not count as a method improvement. Only setting the existing `crossfit` or `superlearner` options of `estfd`, or only reporting the `$Onestep` result it already returns, also does not count.

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
