# Conditionally Identifiable Latent Representation for Multivariate Time Series Forecasting

## 1. Goal

Recover five latent dynamic factors from each multivariate observation matrix
(T=200 time steps, N=20 observed series), maximizing `mcc` (higher is better).
This task compares per-instance factor-recovery estimators on a fixed dynamic
data-generating process. Fit each instance independently using its observations.
You may change the method under the rules in section 6.

## 2. Data

- `data/train/` contains a note only. No cross-instance training split is supplied. Validation is for evaluation and selection only. The task-defined fit on each instance's own observations remains allowed; do not carry fitted state across instances.
- `data/val/input.npz` and `data/val/truth.npz` contain three public validation instances, generated with seeds 2026, 2027, 2028. Inputs contain opaque `id` and `x` of shape (n,200,20); labels contain factors and latent DGP quantities.
- `bench/generator.py` freezes the dynamic DGP: AR(1) factors with rho=0.7 and Laplace innovations, random linear loadings, and AR(1) idiosyncratic noise with alpha=0.2.
- Each supplied matrix is one independent estimation problem. Input contains observations only. Validation labels are only for evaluation and method, hyperparameter or checkpoint selection. Do not fit to them or use them as prediction inputs.
- Do not re-split, relabel, add external data, generate extra simulations, or search for generating seeds. Random algorithmic operations on the supplied observations are permitted.

## 3. Metric and scoring

`mcc` is the mean across instances of the mean absolute Pearson correlation of
five recovered and five true factors after exact maximum-weight one-to-one
matching. Sign and permutation are irrelevant; factor rotations generally are
not. The theoretical upper bound is 1. Constant columns have zero correlation.

| Guard | Enforcement class |
|---|---|
| Exactly one finite (200,5) factor matrix for every supplied id; exact schema, no duplicate or missing id/time keys | scorer (also validator) |
| Same instance gives the same prediction after an opaque-id change or independent replay, with BENCH_SEED fixed | patch audit |
| Recover factors from that instance's observations only; no truth/seed/ID lookup, memorized answers, extra simulations, or cross-instance coupling | patch audit |

The binding worst score is 0 for invalid output; format violations also fail
validation. Observation-only provenance and replay consistency require patch audit.

The frozen scorer implements the mathematical MCC definition with exact matching.
The authors' implementation uses a finite-epsilon auction and sample-standard-
deviation normalization followed by division by T. It is preserved as the
`authors_mcc` diagnostic; it is not the optimization target.

Validation reference: **0.6470676 ± 0.0049638** (population standard deviation across three algorithm seeds).
The judge retrains using three algorithm seeds and compares the mean with its
archived reference. PASS requires relative gain at least
`max(delta, 2 * reference_std / abs(reference_mean))`; task delta is 0.05.

## 4. Baseline

Official code: https://github.com/minkeymouse/ivdfm at commit
`763ad02215e380695710335805f9b5127609a2c2`. `code/model/ivdfm.py` wraps
`dfm-python==0.7.0`; that dependency contains the model, loss, and training loop.
The baseline infers non-Gaussian innovations, propagates them through latent
dynamics and reconstructs the observed series. It is one factor-recovery estimator;
other estimators remain allowed by the research scope.
`code/config/model/ivdfm.yaml` merged with the dynamic iVDFM overrides in
`code/config/experiment/factor_recovery.yaml` defines the baseline, including
200 maximum epochs, batch size 64, AdamW, learning rate 0.002, and the original
early stopping and scheduler.

`code/train.sh` records configuration. `code/predict.sh` fits each instance
independently in a process pool with one numerical thread per worker.
At 8 CPUs, training took 0.72, 0.67, and 0.82 seconds across the reference runs.
The corresponding validation predictions took 23.21, 24.06, and 23.90 seconds.
The declared memory budget is 16 GB (not a measured peak).

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes model/configuration artifacts.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with header `id,t,f1,f2,f3,f4,f5`, exactly 200 rows per input id. See `bench/io_spec.md`.
- Environment: BENCH_SEED, BENCH_CPUS, BENCH_DATA_DIR, BENCH_BENCH_DIR, BENCH_CODE_DIR. Paths passed by the evaluator are absolute.
- Use `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and `bench submit --final`.
- Install dependencies with `uv pip install`; submission freezes them into `code/env.lock`. Only pinned packages from PyPI and the PyTorch CPU index are rebuildable.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- Editable surface: factor-recovery algorithms, learned estimators, training surrogate losses, features and optimization under `code/`, within the fixed dynamic-factor problem and per-instance observation boundary.
- Frozen surface: `bench/`, `data/`, this README, signatures, and prediction format.
- Resource allocation: up to 8 CPUs and 16 GB.
- Honor BENCH_SEED. The baseline maps it to 2025+BENCH_SEED and uses that algorithm seed for each independent instance; generating seeds are a separate random stream.
- No network during train or predict. All needed code must be submitted or installed from env.lock.
- Disqualifying: accessing or guessing hidden data, seed search, truth tables, cached instance answers, memorized paper results, extra generated instances, or interpreting ids/order as features. Use validation labels only for evaluation and method, hyperparameter or checkpoint selection; every returned factor matrix must be computed from its own observations.

## 7. Known pitfalls

Fitting happens in predict, so include that work in your runtime budget. Do not
omit difficult instances, shorten sequences, or truncate the factor dimension.
Use the supplied `--output` path exactly, even if it has no extension. CSV row
order is arbitrary, but id/time keys must be complete and unique. Data are raw;
the baseline applies its own standardization. Alternative models may choose their own preprocessing without modifying supplied observations.
The baseline uses one numerical thread per worker; candidates may choose another allocation within BENCH_CPUS. Use the benchmark entry points; legacy paper scripts generate their own
data and are not valid evaluation entry points.

## Research scope

**Objective.** Recover the five latent dynamic factors independently from each supplied 200x20 observation matrix, maximizing the original optimally matched absolute-Pearson MCC.

**Fixed research objects.** Dynamic DGP with AR(1) factors, Laplace innovations, loading distribution and idiosyncratic-noise process; supplied observation instances; five-factor target and 200-step population, per-instance estimation boundary, exact one-to-one matching and original MCC aggregation. These keep the factor-recovery problem comparable across estimators.

**Editable methods.** Factor-recovery estimators, representation architectures, training surrogate losses and numerical optimizers; internal observation preprocessing and per-instance fitting using its observations alone.

**Forbidden substitutions.** Replacing the generative or latent-factor research model, changing factor dimension/time length or redefining recovery up to arbitrary rotations; cross-instance fitting, additional generated observations, seed inversion, latent truth use or opaque-ID answer lookup. When retaining the supplied iVDFM pipeline, changing only decoder_type=mlp, factor_order=2, num_regimes=7, max_epochs=200, learning_rate=0.002, patience=33, beta_kl=2.96384804579094, dropout=0.09054073499204782, optimizer_weight_decay=0.001826481870844076 or the layer sizes and counts in code/config/experiment/factor_recovery.yaml does not count as a method improvement.

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
