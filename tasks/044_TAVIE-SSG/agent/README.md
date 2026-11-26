# TAVIE-SSG: Student-t joint posterior approximation

## 1. Goal

Research question: how closely can a variational approximation, fitted separately to each dataset, match the true joint posterior of a Student-t regression?

The source paper ("A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods") proposes TAVIE-SSG, a variational method built on tangent minorizers of the log-likelihood, as an alternative to MCMC and to black-box variational inference. Student-t is one of its Type I strongly super-Gaussian (SSG) likelihoods.

This task is Section 4.1 / Figure 4(c), the E1 setting with eight covariates plus an intercept, nu=5, alpha=1. That experiment fits TAVIE-SSG, MFVI, DADVI and ADVI (mean-field and full-rank) to simulated datasets at n=200,500,1000,2000. It compares them by sliced-Wasserstein distance to a PyMC NUTS posterior, which stands in for the true posterior. All methods are compared on the same datasets, in the same joint coordinates, against the same reference. The authors report that TAVIE-SSG, DADVI and MFVI come closer than the ADVI variants, and that for these three the distance falls as n grows.

Here a candidate variational method is compared with TAVIE-SSG in the same way. Approximate the joint posterior of regression coefficients and log precision from a Student-t regression dataset. Improve `sw` (lower is better), the sliced-Wasserstein distance to a fixed PyMC NUTS reference posterior. You may change the variational inference method under the rules below.

## 2. Data

- Model: y_i = x_i'beta + e_i/sqrt(tau2), where e_i follows a Student-t with nu=5 and x_i holds the intercept and eight covariates. tau2 is a precision.
- Simulation: covariates are independent standard normals, tau2_true=3 and beta is drawn from a standard normal in nine dimensions.
- `data/train/README.md`: no cross-instance training dataset. The baseline fits each observation dataset at prediction time.
- `data/val/input.npz` and `data/val/truth.npz`: 40 public validation instances, ten per n=200,500,1000,2000. Inputs contain observed X/y, offsets and opaque ids. Labels contain NUTS samples, true parameters, auxiliary seeds and baseline elapsed times.
- `bench/generator.py`: frozen observation generator, bitwise checked against the validation instances. Public validation beta seed is 123, and all validation instances share that beta; instance seed is 1000+37*r+101*n for r=1..10. The hidden split is held by the judge. Its true parameters cannot be inferred from validation labels.
- Fit each instance independently, using only its own observations and the trained configuration. No pooling across instances, seed search, truth tables, extra simulated observation datasets or outside data. Drawing the output samples from the approximation fitted to the supplied observations is allowed.

## 3. Metric and scoring

`bench/scorer.py` computes the mean absolute difference of sorted projections of two 8000-sample posteriors onto 500 unit-normal directions in the ten-dimensional (beta, log tau2) space. Take the median within each n, then the unweighted mean across n. There is no standardization or subtraction of the NUTS sampling floor.

| Guard / rule | Enforcement |
| --- | --- |
| Required ids, sample dimensions, finite values, positive elapsed times | scorer / validator |
| Recomputed samples agree and measured elapsed times are recorded accurately | judge probe |
| Variational inference only; no MCMC or importance resampling to produce or correct samples | patch audit |
| Only own-instance observations; no hidden-label access, seed search, cached predictions, extra datasets or timing manipulation | patch audit |

Timing is measured from model construction through all 8000 posterior samples; exclude imports, I/O and metric computation. Declare elapsed wall seconds honestly. Invalid output gives Infinity. Elapsed times are diagnostic measurements: the paper also compares runtime, but here it does not enter the score. The score has no coverage or variance-floor gate; collapsing a distribution is penalized through SW rather than a separate threshold. Independent replay verifies timing and prediction provenance.

Measured validation baseline at 32 CPUs, three runs: `sw` mean **0.00536778496182**, seed standard deviation **3.19968e-05**. Mean `sw` by n is about 0.0099 (n=200), 0.0055 (n=500), 0.0036 (n=1000) and 0.0025 (n=2000). The judge uses three algorithm seeds and compares their mean against its reference baseline. PASS requires relative improvement at least max(delta, twice relative reference seed standard deviation); delta is 5%.

## 4. Baseline

Official TAVIE-SSG code: https://github.com/Roy-SR-007/TAVIE-SSG at `0c17df8ebe147bec8d9ca01b4aa92c21fe885ad5`. The model and original convergence loop live in `code/TAVIE/`; `posterior.py` contains the notebook's unmodified normal-gamma sampling and fit wrapper. Prior: m0=0, V0=10I, a0=b0=.005; tolerance 1e-9, maximum 1000 iterations. Preserve the authors' scaling and parameterization in the baseline.

The method bounds the Student-t log-likelihood by a tangent minorizer that is quadratic in the residual, with one auxiliary parameter xi_i per observation. The result is a conjugate normal-gamma approximation: beta given tau2 is N(m, V/tau2) and tau2 is Gamma(shape a/2, rate b/2), with a=a0+n*alpha. Closed-form updates of xi, V, m and b repeat until xi stops changing.

The adapter handles configuration and array I/O, per-instance multiprocessing, BENCH_SEED sampling, and portable paths. `code/student-t_simulations.ipynb` keeps the source of the paper's Student-t experiments; cell 9 is the SW experiment, with the competitor methods and the PyMC model. The notebook's SW experiment uses different priors from its earlier MSE experiment; the paper's MSE values are not this task's target.

Measured at 32 CPUs: train 4.67 to 4.78 s and validation predict 11.49 to 12.14 s per run. Median fit plus sampling per instance is about 0.006 s at n=200 and 0.011 s at n=2000.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes the model configuration.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes a binary NPZ containing id, z and seconds. Exact shapes and timing scope are in `bench/io_spec.md`.
- Retain `code/bench_run.py`'s `fit_sample(X,y,seed,config)` callback for independent replay. It returns all joint draws and includes all instance-dependent computation. It may implement any variational method these rules allow. Replay derives `seed` with the formula in `bench_run.sample_seed`, so recomputed samples agree only if predict uses the same formula.
- Environment: BENCH_SEED, BENCH_CPUS, BENCH_DATA_DIR, BENCH_BENCH_DIR and BENCH_CODE_DIR. Honor the CPU budget and seed; preserve instance-order/subset invariance.
- `bench validate-output FILE` checks a prediction file.
- Dependencies use Python 3.13.18 and PyPI pins in env.lock. Editable, local and git installs cannot be rebuilt.

## 6. Rules

You may replace the variational approximation family, auxiliary approximation loss, optimizer and numerical hyperparameters under code/. The method must be variational inference: fit a parametric approximation by optimization and draw the output samples from it. Do not run MCMC (NUTS, HMC, Gibbs, Metropolis or SMC) or importance resampling to produce or correct samples. The Student-t likelihood, reference posterior target, prior defining that target, output coordinates and evaluation data remain fixed; alternative approximations must target the same posterior. Keep the entry-point arguments, prediction format and replay callback signature. Frozen surfaces are data/, bench/, this README, the scoring protocol and the interface.

The judge provides 32 CPUs and a 32 GB memory allocation; the memory figure is a declared limit, not a measured peak. No runtime network access. Each of three runs must honor BENCH_SEED. Reading or guessing hidden instances/labels, embedding memorized answers, combining instances, import-time fitting, caching instance outputs and falsifying elapsed times are disqualifying. Keep all observations-dependent work inside the replay callback.

## 7. Known pitfalls

A low MSE posterior mean need not give good SW: the metric evaluates the whole joint distribution. tau2 is precision; output its logarithm, not the variance. Do not manually rescale coordinates. The authors' sample routine includes its original covariance jitter.

Use one BLAS thread per independent worker. Dependency import and process startup can dominate elapsed prediction time even when fit+sampling takes milliseconds. When writing NPZ, open the exact output path in binary mode: numpy otherwise appends an extension. Validators accept any instance order but reject missing/duplicate ids and non-finite values. Files ignored by `code/.gitignore` are not submitted.

## Research contract

**Objective:** Approximate the fixed joint Student-t regression posterior in (beta, log tau2), minimizing sliced-Wasserstein distance to its hidden NUTS reference.

**Fixed research objects:** These stay fixed because SW distances compare methods only when each approximates the same posterior, in the same coordinates, against the same reference. The nu=5 Student-t Type-I SSG likelihood, alpha=1, target prior with variance 10 for beta and a0=b0=.005, as in the reference and precision interpretation of tau2 are fixed. The joint ten-dimensional coordinate system, 8000 output samples, 500 fixed projection directions, four n-cell medians and their equal mean remain fixed.

**Editable methods:** Within variational inference, the approximation family, variational auxiliary objective, optimizer and numerical/convergence settings may change while targeting this same posterior. Working approximation distributions need not be the baseline TAVIE family; the target likelihood/prior remain unchanged.

**Forbidden substitutions:** Also do not use accept-reject correction of draws, and do not fit the approximation to MCMC draws. The method must be variational inference: fit a parametric approximation by optimization and draw the output samples from it. Do not run MCMC (NUTS, HMC, Gibbs, Metropolis or SMC) or importance resampling to produce or correct samples. Do not change likelihood, prior or posterior temperature; substitute parameter-estimation MSE for distributional inference; output variance where precision is required; or rescale scoring coordinates. No NUTS reference access during inference, hidden seeds, cached samples, cross-instance pooling or extra simulated datasets. Training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation, is also forbidden.

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
