# A Spectral Confounder Adjustment for Spatial Regression with Multiple Exposures and Outcomes

## 1. Goal
Estimate the nine non-intercept exposure effects on five outcomes from observed spatial X/Y matrices. Improve mean squared error under the original five simulation settings, with 12 replications per setting on a 20×20 grid. The research object is the authors' multivariate spectral model (MSM) posterior with its specified tensor structure, horseshoe priors, L=10 and K=5. Tensor MCMC with 5000 iterations and 1000 burn-in is the baseline posterior computation.

## 2. Data
This synthetic task needs no training split. `data/val/input/manifest.csv` indexes 60 independently materialized observed datasets. Each RDS holds exactly `X` (400×10, intercept followed by nine exposures) and `Y` (400×5); the manifest supplies the grid side. `data/val/labels.csv` contains validation effect truths and grouping information for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable models or learned preprocessing on validation. Fit each instance only from its own observed X/Y, without carrying fitted state to other instances. Public generation seeds are 1–12, reused across the five settings as in the source simulation. The frozen `bench/generator.py`, `generator.R`, original `functions_sim.R`, and loading matrices define the generator. True effects and latent confounders are not included in prediction inputs. Do not resplit or relabel the supplied data.

## 3. Metric and scoring
`mean_mse` is lower-is-better: square the error of each submitted MSM posterior-mean exposure-effect estimate at the first spatial frequency, average over the nine non-intercept exposures, five outcomes, and replications within each setting, then average the five setting MSEs equally. This is the authors' `metrics_over_all3()$MSE_ave1`.

The scorer checks that strict 95% interval coverage (`lower < truth < upper`) is at least 0.93 in **each** setting. Any failed setting yields positive infinity. The validator checks complete, unique effect identifiers, finite numbers, and ordered intervals; the scorer independently rejects invalid outputs with the worst score. The original setting distribution and observed-only access are frozen protocol requirements checked by data integrity and patch review. Measured validation mean_mse is 0.03169814662081416, with standard deviation 0.0018395504481129418 across reference repetitions. All measured validation repetitions passed the per-setting coverage guard.

## 4. Baseline
Official repository: https://github.com/snprim/multivar_spectral_confounder at `30a11dab3b4c320967ee52a6d80bc710b8f6fedd`. The model separates scale-dependent spatial confounding from exposure effects. It transforms X/Y into the lattice-Laplacian eigenbasis. Eigenvalues are ordered from largest to smallest, so the first frequency is the most local scale, where the model assumes confounding vanishes. Smooth spectral effect curves use spline coefficients gamma[l,e,r] = sum_k T_basis[l,k] T_exposure[e,k] T_outcome[r,k]. The shared tensor factors borrow information across exposures and outcomes. Horseshoe shrinkage acts on those factors; spatial latent factors and outcome-specific noise complete the likelihood. The fixed target is defined in `code/simulation-code/functions_sim.R`, in `MCMC_model4_8_v7`. Keeping this likelihood, tensor and prior structure fixed makes the statistical target identical across posterior algorithms. `code/reproduce/sim_driver.R` computes MSM, Naive, Spatial+80, Spatial+40 and USM posterior summaries. The supplied baseline runs all five fits and submits MSM summaries. Alternative inference and approximation algorithms for the fixed MSM posterior are allowed; candidates need not run the unscored comparison fits. Replacing MSM, its priors or the requested posterior mean/quantiles changes the research object and is not permitted. These target semantics require implementation audit in addition to output validation. At 32 CPUs, measured training-stage recording took 0.12 seconds in each repetition. Validation prediction took 1169.91, 1073.37 and 1145.39 seconds across reference repetitions.

## 5. Interface (frozen)
`bash code/train.sh --out DIR` creates the model output directory. There is no cross-dataset training. `bash code/predict.sh --model DIR --input PATH --output FILE` performs fresh model fitting and writes effect summaries. `BENCH_SEED` controls fitting randomness; `BENCH_CPUS` controls independent fitting workers, each using one BLAS thread. `BENCH_DATA_DIR` and `BENCH_BENCH_DIR` provide the frozen public data and tools. The supplied read-only runtime is an R 4.4.3 snapshot. Its dependency closure and `env.lock` are fixed. `bench/io_spec.md` defines the CSV contract.

## 6. Rules
You may modify code and methods under `code/` while preserving the interface and frozen protocol. `data/`, `bench/`, this README, the runtime snapshot, and the output contract are frozen. Resources are 32 CPUs and 144 GB memory. No network, hidden-data access, hard-coded predictions, or cached fitted outputs are permitted.

## 7. Known pitfalls
Keep strict interval endpoints and exclude the intercept when computing the metric. The supplied observations are fixed; fitting randomness is controlled separately through `BENCH_SEED`. Numerical validity does not establish that submitted summaries target the required posterior.

## Research contract

**Objective:** Improve posterior computation for the specified multivariate spectral confounder model, evaluated through its first-frequency exposure-effect posterior means and 95% credible intervals.

**Fixed research objects:** The MSM likelihood, shared tensor spectral effect parameterization, horseshoe/hierarchical priors and hyperparameters in functions_sim.R, L=10 and K=5 define the target posterior. First-frequency posterior means and equal-tailed 2.5%/97.5% quantiles, nine non-intercept exposures/five outcomes, fixed five settings, equal MSE and per-setting strict coverage >=.93 remain fixed.

**Editable methods:** Posterior sampling, variational or other approximation, numerical parameterization and optimization may change while targeting the same posterior and summaries. Iterations, burn-in and scheduling may be set as part of a new posterior algorithm or approximation. Auxiliary approximation losses are solver choices; unscored Naive/Spatial+/USM comparison fits may be omitted.

**Forbidden substitutions:** Do not replace the MSM likelihood/tensor/prior model with an ordinary regression or change the posterior quantities to generic estimates/intervals. Do not change basis/rank model controls L/K, use hidden confounders/effects, pool independent cases or alter observed X/Y and scoring. When retaining the supplied Gibbs/Metropolis sampler, changing only iters=5000, burn=1000, MH=0.1 or its warm-up adaptation settings (att>50, acceptance bounds 0.3 and 0.5, multipliers 0.8 and 1.2) does not count as a method improvement.

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
