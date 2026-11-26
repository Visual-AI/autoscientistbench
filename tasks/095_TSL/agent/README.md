# Interpretable low-rank regression on California housing

## 1. Goal
Improve raw-scale prediction RMSE for district median house value from eight
census features, while retaining the paper's low-separation-rank interpretability
budget. A candidate predictor is a sum of at most **four rank-one products of
univariate functions of the raw features**. A nonzero constant offset counts as
one product. Models, losses, optimizers, factor functions and training methods
may be replaced within this scientific class.

This budget comes from the paper's interpretable R<=2 experimental condition:
each original TSL stage has two products, giving total separation rank <=4.
Candidates need not use TSL, two stages, ordered positive/
negative pairs, positivity, or the baseline step-function factors. Arbitrary
univariate functions, including learned neural factors, are allowed.

## 2. Data
- `data/train/train.csv`: 13,209 fixed training rows, with columns
  `id,longitude,latitude,housingMedianAge,totalRooms,totalBedrooms,population,households,medianIncome,y`.
  Inputs are the original raw OpenML values; y is median house value in USD.
- `data/val/input.csv` and `data/val/labels.csv`: 3,303 validation rows. Validation
  was carved from the paper's training split with fixed seed 20261029.
- Held-out examples and labels are unavailable in the workspace.
  Input IDs are bookkeeping only.
- Use the supplied training data for fitting and validation for model selection.
  Do not replace, resplit, relabel or augment inputs, import outside data, download
  the public source dataset, load it from an installed package, or reconstruct held-out rows. Model-internal transforms
  are allowed inside the univariate factors, without modifying supplied files.

## 3. Metric and guards
The frozen scorer computes `sqrt(mean((y-pred)^2))` on the original USD scale.
The output must cover every input ID exactly once with finite predictions.

The model probe reconstructs predictions from the saved separable representation
and requires maximum relative error
`abs(pred-reconstructed)/max(1,abs(pred)) < 1e-6`.
It enforces a total budget of four products, counting a nonzero offset as one.
Every factor can use only one raw feature value and saved training artifacts.
No candidate factor may inspect other columns, row IDs or other evaluation rows.
Executable factors are evaluated in separate isolated processes on one scalar at
a time; the trusted probe multiplies factors and sums terms itself.

Run `python bench/check_model.py MODEL_DIR --input INPUT --pred PRED` separately
after a development evaluation. Passing this model check and a data/training
provenance audit is required for acceptance. A failed structural or reconstruction
check disqualifies the result even if its RMSE is finite.

Baseline validation RMSE is 54,546.16 USD, with standard deviation 501.99 over
three seeds in the latest reference. The relative improvement threshold is
max(0.005, twice reference seed standard deviation divided by its mean).

## 4. Baseline
The supplied baseline is official `tensorsl==0.3.0`, repository
https://github.com/jyliuu/TSL at `b66f7bbd37b18e00faed97c2ff6994374a7ecef8`.
Its released California housing configuration uses two stages, 200 trees, 64
iterations and bagging. The original Rust/PyO3 implementation, factor exporter,
loss and training loop are retained in `code/`. Those are baseline choices.

The adapter reads materialized data, honors BENCH_SEED/BENCH_CPUS and saves
`model.bin`, `model_card.json` and training diagnostics. The baseline is deterministic
for a fixed seed. In the latest four-CPU reference, training took 18.70, 19.15 and
18.60 seconds for seeds 1, 2 and 3. Validation prediction took 2.87, 2.47 and
2.52 seconds, respectively. Built-in step and linear factors remain efficient
options; executable factors may have substantial startup overhead.

The current dependency setup uses a PyPI wheel; no Rust toolchain is supplied.
Alternative Python implementations or reproducible pinned packages may implement
any model in the allowed separable class. Editing unused Rust source alone does
not change the installed predictor.

## 5. Interface
- `bash code/train.sh --out DIR` writes all fitted artifacts, including
  `model_card.json`. The generic card describes up to four separable terms;
  the original staged card remains supported. See `bench/io_spec.md`.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes `id,pred` CSV.
- Honor BENCH_SEED, BENCH_CPUS, BENCH_DATA_DIR and BENCH_BENCH_DIR. The judge uses three seeds, four CPUs and 4 GB memory. No network is available during execution.
- Use `bench status`, `bench eval-dev`, `bench validate-output FILE` and
  `bench submit --final` for development and submission.
- Dependencies must be reproducible exact PyPI pins in `code/env.lock`.
  Editable, Git and local-path installations are unsupported.

## 6. Rules and pitfalls
Everything under `code/` may change while preserving the fixed inputs, scientific
rank budget, reconstruction check and RMSE metric. `data/`, `bench/`, README and
script/output signatures are frozen. No held-out information, memorized paper
predictions, misleading cards or prediction-time model changes are allowed.

The model card must describe the predictor actually fitted and used by predict.
Univariate factors may be nonlinear, signed or learned with arbitrary losses.
Multiple factors of the same raw coordinate still form a univariate function and
are permitted within a product. A constant can be a factor-free product or the
intercept; each nonzero standalone constant term consumes one rank slot.

For executable factors, train exports a frozen `factors.zip` artifact; the checker
supplies only the selected scalar and function identity, not an evaluation row.
Each invocation starts fresh. Execution cost depends on the number of unique
factor inputs and model startup overhead.

Keep full-precision model-card values and the original right-bin tie convention
when using step factors. The baseline's random bagging affects RMSE across seeds.
Training uses fewer rows than the original paper refit because validation is held
out; compare against benchmark references. Targets retain their original upper
censoring at 500,001 USD, and the RMSE formula is unchanged.

## Research contract

**Objective:** Improve raw-USD house-value RMSE within the paper-supported low-separation-rank interpretable regression class.

**Fixed research objects:** The predictor is a sum of at most four rank-one products of univariate functions of the eight original raw features; a nonzero standalone constant consumes one term. Original target units/censoring, supplied fixed train/validation/test splits, pointwise raw-feature factor access, reconstruction tolerance<1e-6 and RMSE remain fixed.

**Editable methods:** Training algorithm, surrogate loss, optimizer and arbitrary univariate factors including neural functions may change within the same four-product capacity class. Two-stage TSL, ordered signs, positive factors and step-function parameterization are baseline constructions rather than the scientific rank budget.

**Forbidden substitutions:** Do not replace the four-product separable model with a generic neural/tree predictor, add a fifth term/unaccounted intercept, or let one factor inspect multiple raw coordinates or row identity. No external/held-out data, changed raw features/target units or dishonest structure/reconstruction cards. When retaining the supplied TSL pipeline, changing only n_trees=200, n_iter=64, split_try=10, min_interval_samples=56 or the other released `best_params` values in `code/configs/california_housing_tsl_r2.json` does not count as a method improvement.

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
