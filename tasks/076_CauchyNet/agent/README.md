# CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions

The judge retrains your code from scratch and scores a hidden evaluation split.

## 1. Goal

Learn a compact scalar regression function from observed coordinate/value pairs.
This is the released one-dimensional gap-filling experiment: observations leave
regions of the input domain unobserved, and evaluation measures reconstruction
in those regions. Improve `mae` (lower is better), on the fixed supplied observations with at most 128 real trainable parameters per fitted model. The metric averages the errors of ten independently
trained models, with fixed model seeds 0–9.

## 2. Data

- `data/train/train.csv`: 200 observations with columns `id,x,y`, natural units,
  represented exactly as float32 values.
- `data/val/input.csv` and `data/val/labels.csv`: 50 validation observations,
  columns `id,x` and `id,y`. These are public and used for checkpoint selection.
- The evaluation split is held by the judge. Training and validation cover
  the observed regions; evaluation covers withheld regions. Validation and
  evaluation therefore have different coordinate distributions.
- Repeated boundary coordinates reflect the original sampling and must remain.
  Do not re-split, relabel, regenerate targets, or bring in external data.
  IDs are opaque and carry no predictive meaning.

## 3. Metric and scoring

`bench/scorer.py` computes float32 absolute errors and a mean for each model,
then averages the ten MAEs in float64. Averaging predictions before computing
errors changes the metric and is forbidden. Any violated guard yields `Infinity`.

| Guard | Enforcement class |
|---|---|
| Ten finite prediction columns with valid parameter declarations | scorer and validator |
| Each declared parameter count is at most 128 real parameters | scorer |
| Ten independently fitted models | patch audit |
| Fitted artifacts reproduce predictions; diagnostics are truthful | judge probe (`bench/check_model.py`) and patch audit |
| Each prediction depends only on its own coordinate and fitted model | judge probe (subset replay) and patch audit |
| No hidden-data reconstruction, memorized answers, or learned constants disguised as buffers/program text | patch audit |

Complex trainable tensor elements count as two real parameters; real elements
count as one. Genuine fixed, data-independent buffers are excluded. The
structural probe must pass in addition to CSV scoring; a declaration alone is
not proof of compliance.

Baseline validation MAE is **0.0008921614033170044**, outer-run σ = 0.
`bench status` also reports the archived validation metric. The fixed ten
model-seed suite is evaluated as one deterministic outer run. PASS requires a
relative improvement of at least **1%** (`delta=0.01`), with the harness's noise
threshold applied when larger. The observed zero outer-run variance describes
the fixed suite, not uncertainty over new datasets or model seeds.

## 4. Baseline

The method is CauchyNet with fixed elliptical poles, 64 complex coefficients,
128 real trainable parameters, radii (2.5, 0.4), Adam lr=0.05, 3000 epochs,
batch size 32, gradient clipping 1.0, and StepLR(2000, 0.5). Only the real output
is scored. The imaginary penalty is zero. The lowest validation MSE checkpoint
is retained, using the authors' equal weighting of validation batches.

The baseline model and training loop are in
`code/experiments/best_config_gap_filling.py`. The training entry point loads
materialized observations, selects a checkpoint, and exports all ten models.
Training uses up to 8 independent single-threaded processes; prediction uses
at most 8 threads. Measured on 8 CPUs: training took **39.18 s** and
validation prediction took **2.97 s**. The task memory budget is 8 GB.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: trains from scratch and saves all ten models.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: writes the CSV
  specified in `bench/io_spec.md`, even when FILE has no extension.
- `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`: data, frozen helpers and
  code locations. Paths passed by the harness are absolute.
- `BENCH_CPUS`: total CPU budget. `BENCH_SEED`: execution seed. Because the
  metric fixes the ten model seeds, this controls their independent submission
  order, not their initialization seeds. Results must be invariant to this order.
- Supply `code/model_api.py` with `predict(model_dir,x)` returning an N by 10 prediction array, or export `model0.ts` through `model9.ts` using the legacy TorchScript interface. Generic methods must also export `model0.parameters.npz` through `model9.parameters.npz`, containing all learned scalar arrays. The probe recounts these safe numeric manifests and replays fitted state.
- `bench status`, `bench eval-dev`, `bench validate-output FILE`, and
  `bench submit --final` are the development commands.
- Dependencies belong in `code/env.lock`: only pinned `name==version` packages
  from PyPI or the PyTorch CPU index. The Python version is 3.11.

## Research contract

**Objective.** Minimize mean gap-filling MAE across ten independently fitted
models under the fixed capacity budget.

**Fixed research objects.** Use the supplied observations and split design,
the fixed model seeds, and at most 128 real learned parameters per model.
Compute each model's MAE in float32 and average those errors in float64.
`data/`, `bench/`, this README, script signatures and prediction schema are frozen.
The capacity budget counts every learned scalar used by each model; a complex
scalar counts twice. Genuine fixed mathematical constants are not learned capacity.
The scorer checks reported counts and the model probe recounts exported parameter
evidence; completeness still requires source review. The judge runs the full
fixed seed suite from scratch. CPU budget is 8 and memory budget is 8 GB.

**Editable methods.** You may change the model, loss, features, training procedure,
or the hyperparameters of a changed method under `code/`, subject to the fixed capacity, seed, data
and output contract. CauchyNet is the baseline method; other compact function
estimators remain allowed within the same scientific comparison.

**Forbidden substitutions.** Do not exceed the capacity budget, replace the
per-model errors with an ensemble error, or disguise learned constants as fixed
buffers or frozen parameters. No network is available during training or prediction.
Use only the supplied training observations to fit models; validation is for
evaluation and method, hyperparameter or checkpoint selection. Do not reconstruct
the hidden split, use the published target formula as an oracle, embed known
answers, or condition predictions on IDs, input order, other rows, or split identity. When retaining the supplied CauchyNet pipeline, changing only lr=0.05, epochs=3000, batch_size=32, hidden_size=64, r_re=2.5, r_im=0.4, imag_pen=0.0 or StepLR(2000, 0.5) does not count as a method improvement.

## 7. Known pitfalls

Validation excludes the missing regions, so validation improvements need not
transfer to gap filling. Keep ten independent predictions rather than an
ensemble. Complex parameters have twice their tensor element count. TorchScript
exports must retain all learned parameters and work with varying input batch
sizes. Keep required files committed: the judge uses a clean clone.

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
