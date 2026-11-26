# Calibrated Test-Time Guidance for Bayesian Inference

## 1. Goal
Improve posterior sampling for the fixed SLCP prior/likelihood problem; calibrated Bayesian guidance (CBG) is the baseline method.
The input is an observed dataset and its prior/likelihood family; the output is
10,000 posterior parameter samples. The objective is lower `c2st`, the accuracy
of a classifier distinguishing your samples from independently held reference
posterior samples. Accuracy near 0.5 indicates indistinguishable distributions.
The submitted method must also preserve inference quality on Gaussian linear,
uniform-prior Gaussian linear, Gaussian mixture, and two moons problems.
You may change the method under `code/` subject to the frozen protocol below.

## 2. Data
- `data/train/` explains the absence of a training split. The baseline has no training. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Per-instance inference may use its current observation, without carrying state fitted on one case into other independent cases.
- `data/val/input/` contains 25 development instances, five per family; each NPZ
  holds `task`, `dim`, and a float32 `observation`. Filenames are opaque ids.
- `data/val/labels.npz` contains 10,000 reference samples per instance and a JSON
  `__meta__` field describing generation and the frozen development guard baselines.
- `bench/generator.py` is the frozen sbibm forward simulator. Development
  observations use public seed 20261030, with per-instance seeds in label metadata.
  They are newly simulated observations independent of the held-out evaluation.
  You may generate additional synthetic development observations using this simulator.
- The hidden evaluation split is held by the judge. Its observations follow the
  same inference families and preprocessing; source identities are not supplied.
- `data/runtime/simple/` is a frozen index of the exact baseline dependency wheels.
  General package/data download sites are unavailable.
- Do not re-split, relabel, fetch external datasets, or use label-derived lookup
  tables. Prediction may only use the current instance observation, the declared
  model family, and a permitted inference algorithm.

## 3. Metric and scoring
`bench/scorer.py` computes `c2st` (lower is better): the mean SLCP classifier
accuracy using the authors' sbibm MLP, five-fold cross-validation, and classifier
seed 1. Each classifier uses 10,000 generated and 10,000 reference samples,
reference-only z-scoring, two ReLU hidden layers of width 10 times parameter
dimension, Adam, and at most 10,000 optimization iterations. See `bench/io_spec.md`.

| Guard | Enforcement |
|---|---|
| Mean C2ST across the four other families must be at most 0.5234 | scorer |
| Each other-family mean must be at most its frozen baseline plus 0.03 | scorer |
| Reported likelihood evaluations per output sample must be at most 100,000 | scorer |
| Reported counts must equal actual frozen-interface evaluations | judge replay probe and patch audit |
| A common inference algorithm uses only supplied observations, prior/likelihood structure and counted oracle evaluations | patch audit |
| Only the current instance observation; no memorized posterior or source-identity lookup | patch audit and subset replay probe |

A protocol violation has worst score 1.0; the scoring interface reports `Infinity`
to make the harness reject that submission, with `protocol_worst: 1.0` in diagnostics.
The budget counts every parameter vector evaluated, including rejected candidates.
Selecting the input's declared prior and likelihood is allowed. Solver selection or tuning keyed to family identity is disallowed. A common adaptive
algorithm may respond to observed likelihood geometry, dimension, prior support
and other permitted problem structure; this is not answer lookup or family-specific tuning. Shape, id, duplicate-key,
and finite-value checks are also mandatory.

The measured development baseline is `c2st = 0.680029988` (one prescribed
random stream; archived standard deviation 0). The development other-family mean
is 0.512940001; all guards pass. The archived standard deviation does not estimate
variation across random streams. `bench status` also exposes this baseline.
PASS requires at least 1% relative improvement against the hidden reference,
subject to all guards.

## 4. Baseline
Official code: https://github.com/mandt-lab/Calibrated-Guidance at
`c339d01413265c67fe4d8f69c6789a7a5e26bacc`. `code/` retains the guidance library and
SBI likelihood/prior definitions. The clean initial snapshot excludes datasets,
result artifacts, and unrelated GPU experiments.

The baseline is gradient-free REINFORCE CBG with 100 time points, 1,000 candidates
per step, and 10,000 samples. The original time schedule performs 99 Euler
updates, giving 99,000 likelihood evaluations per output sample. The original
numerics, clipping, random draws, and update equations are retained.
`code/bench_predict.py` replaces dataset-index loaders with input NPZ loading and
saves samples and evaluation counts. `code/train.sh` writes a no-training marker.
Independent instances run in separate processes, each using `BENCH_CPUS` divided by the instances still unfinished as numerical threads.
On 64 CPUs, the latest measured train time is 0.07 s and development
prediction time is 1448.53 s for the prescribed random stream. The declared
memory allocation is 240 GB; it is a budget, not a measured peak.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` creates the model directory and its marker.
- `bash code/predict.sh --model DIR --input PATH --output FILE` consumes an input
  directory and writes exactly one NPZ at `FILE`; its keys and shapes are frozen.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, and
  `BENCH_CODE_DIR`. The baseline maps `BENCH_SEED=s` to sampler seed `s-1`.
- Load `likelihood_env.py` by absolute path from `BENCH_BENCH_DIR`. Use its public
  `Likelihood` interface for every likelihood evaluation and report `finish()`.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, and
  `bench submit [--final] [-m MSG]` provide the standard development workflow.
- Dependencies are pinned in `code/env.lock`. For an offline reinstall, use
  `uv pip install --index-url "file://$BENCH_DATA_DIR/runtime/simple" PACKAGE`.
  PyTorch CPU wheels are also available from the configured PyTorch index.
  Local/editable/git requirement lines are not valid lock entries. General PyPI
  downloads are deliberately unavailable; the frozen mirror supports the baseline
  numerical stack. No network is available during train/predict.

## 6. Rules
Inference algorithms, posterior approximation families, auxiliary approximation
losses and optimizers under `code/` may change. Numerical hyperparameters may
change as part of such a method change. The
five prior/likelihood targets, their parameter coordinates and the common
task-agnostic inference requirement remain fixed. Preserve the entry points,
counted likelihood access and output format. `data/`, `bench/`, this README,
and the scoring protocol are frozen. Do not modify, shadow, copy to bypass, or
inspect the private state of the frozen likelihood interface.

The evaluation uses one prescribed random stream. Every run must honor
`BENCH_SEED`; stochastic candidate variance is not averaged across extra seeds.
The CPU budget is 64 and the declared
memory limit is 240 GB. General external datasets and pretrained inference answers
are forbidden.

Disqualifying behavior includes reconstructing source observation identities,
installing packages carrying evaluation data, reading or memorizing reference
posteriors, looking up answers from development labels, falsifying evaluation
counts, bypassing the likelihood interface, and specializing the algorithm to
individual task families or known observations.

## 7. Known pitfalls
- Prediction is expensive: each instance evaluates nearly a billion candidate
  parameter vectors. Keep total threads within `BENCH_CPUS` to avoid oversubscription.
- C2ST is a real learned classifier, not a distance computed from summary moments;
  its frozen scoring step can take several minutes after sampling finishes.
- Different families have different dimensions. Do not pad or rescale submitted
  parameters; the scorer applies the original reference-only normalization.
- `PYTHONSAFEPATH=1` is set. Keep the entry point's explicit package path export.
- NumPy appends `.npz` when passed a filename. Write through an open binary file
  handle so an extensionless output path is honored.
- Use the supplied numeric observation and the frozen likelihood interface;
  external datasets are neither needed nor permitted.
- New development draws include sharply concentrated posteriors; inspect the
  per-instance diagnostics as well as the mean.
- The public forward model describes a distribution, not an answer table.
  Exploiting observation identities or memorized reference samples is a violation.

## Research contract

**Objective:** Improve sampling of the fixed SLCP posterior while retaining a common inference algorithm and posterior quality across the four specified companion families. Each target is the posterior p(theta | y) proportional to p(theta) p(y | theta), using the supplied prior, likelihood and current observation.

**Fixed research objects:** The five supplied Bayesian prior/likelihood targets (Gaussian linear, uniform-prior Gaussian linear, SLCP, Gaussian mixture, two moons), parameter coordinates and current observed instance are fixed. 10000 posterior samples, original C2ST classifier/normalization, SLCP mean and companion-family guards, common task-agnostic solver and <=100000 counted likelihood evaluations per sample remain fixed.

**Editable methods:** Common posterior solvers, approximation families, auxiliary approximation losses, numerical integration/sampling and optimizers may change. Generic adaptive behavior may respond to permitted observations, prior support, dimensions and oracle geometry; selecting the declared target prior/likelihood is legitimate. Additional development observations may use the already permitted frozen simulator; no new evaluation inputs or target distributions may be substituted.

**Forbidden substitutions:** Do not modify priors/likelihoods or posterior temperature, replace posterior sampling with predictive fitting, use family-ID-specific solver/tuning tables, or retrieve known observations/posteriors. No reference sample fitting/lookup, uncounted likelihood copies, forged counts, hidden package datasets or cross-instance evaluation information. When retaining the supplied REINFORCE CBG sampler, changing only N=100 time points, K=1000 candidates per step or the uniform-prior start offset 1e-3 does not count as a method improvement.

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
