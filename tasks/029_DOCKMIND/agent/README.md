# Fit virtual-screening hit-rate curves

## 1. Goal
Fit the published experimental hit-rate curves of AmpC, D4 and Sigma2. Improve `mse` (lower is better). The bounded six-parameter DOCKMIND statistical model is the fixed research object. You may replace the optimizer or fitting algorithm.

This is a fixed-instance fitting task. Fit only to `data/train/train.csv`. The public validation files repeat those observations for evaluation and selection only. This task does not measure generalization to held-out molecules.

## 2. Data
`data/train/train.csv` contains the original 93 observations: 36 AmpC, 24 D4 and 33 Sigma2 points, with columns `target,pprop,hit_rate_mean,hit_rate_low,hit_rate_high,definition`. pProp is minus log10 of selected library fraction; definition is the pKi threshold; hit rates are fractions. Confidence intervals may inform a method, while the objective uses the mean.

`data/val/input.csv` and `data/val/labels.csv` repeat this same instance for five fits. Preserve all observations, target definitions and split descriptors. Internal preprocessing is editable; supplied files and evaluated targets are fixed. Do not add outside data or embed published fitted predictions.

## 3. Metric and scoring
The scorer computes MSE within each target, averages targets equally, then averages fit runs 0–4. It reconstructs predictions from each submitted six-parameter certificate using the frozen HitRateModel; self-reported predictions must agree. All ordered input IDs, targets and runs must be represented with finite values. Invalid outputs are rejected. Each randomized fit uses its literal run seed; deterministic fits may legitimately agree. Do not copy a selected best random fit across other runs.

The model family, published bounds and frozen numerical forward map are required. The baseline genetic algorithm is a replaceable solution choice. PASS requires relative improvement of at least `delta=0.01` (1%); seed noise can raise that threshold. Measured validation MSE is 0.0013872167540065763, with standard deviation 0.0 across reference repetitions. Training took 715.17, 715.45 and 719.12 seconds. Validation prediction took 2.17 seconds in each repetition.

## 4. Baseline
Official DOCKMIND commit `9cd2a9e47a375fa27229b0127bdd5fb075d503f3` supplies the six-parameter model. `rho` links standardized docking score to affinity; `exp_mean` and `exp_std` describe the pKi distribution. `artifact_freq`, `artifact_mean` and `artifact_std` describe a separate docking-score population whose affinity is independent of score. This component can explain the decline in hit rate among highly ranked molecules. Parameter bounds and certificate format are in `bench/io_spec.md`. The baseline uses the authors' inverse-CDF lookup and DEAP GA with 100 generations, 5000 individuals, crossover/mutation probabilities 0.4 and tournament size 3. These settings describe the solver; the statistical model and its bounds define the fixed fitting problem. Supplied source does not include fitted answers. Allocation: 16 CPUs and 8 GB memory.

## 5. Interface
`bash code/train.sh --out DIR` saves fitted artifacts. `bash code/predict.sh --model DIR --input PATH --output FILE` writes predictions specified in `bench/io_spec.md`.

`BENCH_SEED` controls the execution order of fits 0–4, each initialized with its literal run seed. Honor `BENCH_CPUS`; resolve mounted resources using `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`. Use `bench status`, `bench eval-dev`, `bench validate-output` and `bench submit --final`. Submission freezes `code/env.lock`; training and prediction have no network.

## 6. Rules
Edit the parameter-fitting procedure under `code/`, including its optimizer, search strategy and internal numerical parameterization. Keep the six-parameter statistical model, bounds, supplied observations, hit definitions, five-run aggregation, scoring formula and entry-point signatures fixed. Recompute solutions from designated training observations. Use validation only for evaluation and method, hyperparameter or checkpoint selection. Private evaluation labels, cached fitted outputs and hard-coded published solutions are forbidden.

## 7. Known pitfalls
Targets are equally weighted despite unequal sizes. The five-run mean is neither the best restart nor MSE of an averaged curve. Performance on a public fit instance is not out-of-sample evidence. Seed metadata cannot prove independent fitting; code and execution provenance remain auditable.

## Research scope

**Objective.** Fit the fixed bounded six-parameter DOCKMIND statistical model to the 93 published observations, minimizing the original target-balanced five-fit MSE.

**Fixed research objects.** the bivariate-normal docking-score/affinity model plus Gaussian artifact component and its six published parameter bounds; one parameter vector per target and run, fixed HitRateModel numerical forward map, all 93 observations and original definitions; equal weighting of AmpC, D4, Sigma2 and literal fit seeds 0..4; fixed-instance in-sample evaluation.

**Editable methods.** the optimizer, parameterization, search strategy, numerical fitting procedure and implementation; surrogate objectives or approximations used internally to find better feasible parameters, while final scoring uses the original MSE and forward model.

**Forbidden substitutions.** replacing the statistical research model with a free predictor, interpolation table or one parameter set per observation; changing parameter bounds, hit definitions, target weights or five-run averaging; submitting measured hit rates directly or reusing published fitted answers instead of recomputing a solution. When retaining the supplied DEAP GA, changing only generations=100, population_size=5000, cxpb=0.4, mutpb=0.4, cxBlend alpha=0.5, mutGaussian sigma=1 or indpb=0.2, or tournament size=3 does not count as a method improvement. Parameters found during development must be recomputed by the submitted code, not hard-coded.

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
