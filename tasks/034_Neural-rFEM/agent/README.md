# Neural r-adaptive meshes for the multi-material L-shaped domain

The judge retrains your code from scratch and compares hidden evaluation error with
an archived baseline. The statement and interfaces below are binding.

## 1. Goal
Predict an r-adaptive tensor mesh from two positive material coefficients for
`-div(sigma * grad(u)) = 1`. The domain is the unit square with its lower-right
quadrant removed, with homogeneous Dirichlet conditions on its boundary. The
material coefficient is s1 in the bottom-left quadrant, 1 in the top-left quadrant,
and s2 in the top-right quadrant. Each axis has 10 elements and includes the
material interface at 0.5. Learn where to move the fixed number of nodes as the
material coefficients change. Improve `mean_relative_ritz_error` (lower is better),
evaluated using conforming Q1 FEM on your predicted meshes. You may change the
method within the rules in Section 6.

## 2. Data
- `data/train/input.csv` and `labels.csv`: 224 coefficient pairs, with original integer
  `id,s1,s2`; labels additionally contain `J_ref`, the authors' reference Ritz energy.
- `data/val/input.csv` and `labels.csv`: 56 public validation pairs. These were carved
  out of the original training split with a fixed random draw, retaining four corner
  coefficient pairs in training. All files are sorted by id.
- The baseline fits on train only and takes its last iterate. Candidates must also fit
  on train only, including final models; validation may be used for selection.
- Do not reconstruct missing grid rows, re-split or relabel the provided data, or add
  external data. No runtime downloads are available.

## 3. Metric and scoring
The frozen `bench/scorer.py` computes mean `sqrt(abs(J_ref-J_mesh)/abs(J_ref))`.
J_mesh is the Ritz energy of the finite-element solution on the submitted mesh.
J_ref is the supplied high-resolution reference energy.
It accepts meshes, not self-reported energies. The exact format is in `bench/io_spec.md`.
A violated guard has score-rule value 1.0; the main metric becomes Infinity so the
harness treats the submission as invalid.

| Guard | Enforcement class |
|---|---|
| Exactly 11 nodes per axis, strictly sorted, finite, within [0,1], including 0, 0.5 and 1 | scorer and validator |
| Exact input ids/order and no missing or duplicate rows | scorer and validator |
| Finite, nonnegative timing fields; identical training time on all rows | scorer and validator |
| No prediction-time FEM solves | scorer checks zero declaration; actual execution requires patch audit |
| Fixed trained mesh rule; no fresh per-input PDE/mesh fitting, evaluation-data reconstruction or memorized outputs | patch audit |

Prediction applies a reusable coefficient-to-mesh rule learned on the designated training data. FEM solves and mesh optimization may be used during training, but prediction must not solve or optimize a fresh PDE/mesh instance. The trusted scorer independently solves FEM to evaluate the submitted mesh. A zero solve-count declaration alone cannot prove this boundary or lawful provenance; source review is also required.

Baseline validation error is **0.195739800927**, population standard deviation
**0.005437973218**, across three reference runs. Delta is **5.8%**. The judge
uses three independent seeds and compares mean error with the archived reference.
PASS requires relative improvement at least `max(delta, 2*baseline_std/baseline_mean)`;
`bench status` reports the task's delta and baseline validation value.

## 4. Baseline
Official code: https://github.com/ManuelSanchezUribe/rFEM-NN at
`fd1c2e0936d8c10ccdfb0277cf8a05451caff786`. `code/` starts with a minimal source
snapshot of the FEM modules. The reproduced notebook training driver is
`code/train_lshape_param.py`, with equivalent sparse assembly in `code/fast_fem.py`.
The model is [2,10,10,18], tanh, LeCun initialization, softmax and cumulative sums.
It runs 1,400,000 Adam steps, batch size 1, constant learning rate 0.01, preserving
the authors' constant optimizer step argument and normalization grid.

The baseline saves the last iterate and reads the seed and CPU budget from the
environment. Prediction uses the endpoint serialization rule described below.
With 32 CPUs, training took 2127.82, 2116.08, and 2115.41 seconds across reference
runs. Validation prediction took 2.17, 2.12, and 2.12 seconds, respectively. The
memory limit is 8 GB. Tiny neural predictions use one CPU within that budget.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR`: train from scratch and save model artifacts in DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: predict meshes as CSV.
- `BENCH_SEED` controls random initialization and sampling; `BENCH_CPUS` limits threads.
  `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR` locate the mounted directories.
- `bench status`, `bench eval-dev --seed S`, `bench validate-output FILE`, and
  `bench submit --final` provide the development workflow.
- Dependencies are pinned in `code/env.lock`, Python 3.12, from PyPI. Only
  `name==version` packages can be rebuilt. Install development dependencies with uv;
  `bench submit` freezes the environment.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

The learned coefficient-to-mesh estimator, training surrogate loss and optimization
under `code/` may change within the parametric research scope below; the original
neural architecture and training hyperparameters are baseline choices, subject to the limit under Forbidden substitutions. `data/`, `bench/`, `README.md` and the interfaces
are frozen. Train and predict have no network. Honor the supplied seed on every run.

The task budget is 32 CPUs and 8 GB memory. Never read or infer the hidden split, embed paper result tables, hard-code predictions or falsify diagnostic declarations.

## 7. Known pitfalls
The baseline runs JAX with float64 enabled; alternative methods must serialize meshes at sufficient precision to meet the frozen geometry checks. Softmax endpoint rounding can produce 1 plus a
few machine epsilons: the baseline only canonicalizes the known endpoint within eight
float64 epsilons and never moves interior nodes. Coincident nodes are inadmissible.
The normalization grid in the original training loss differs from the returned mesh;
this is an intentional preservation of the reproduced implementation.

`PYTHONSAFEPATH=1` means scripts must explicitly set their import path. All interface
paths are absolute. Write only to model/output directories or temporary storage.
CSV serialization must preserve full precision. Report diagnostic timings for the work actually performed, including setup, initialization and compilation.

## Research scope

**Objective.** Learn a reusable coefficient-to-mesh predictor for the fixed parametric multi-material L-shaped elliptic PDE, minimizing the original Q1 Ritz-energy relative error.

**Fixed research objects.** Fixed L-shaped domain, material coefficient meanings, forcing, homogeneous Dirichlet conditions and supplied coefficient split; conforming Q1 evaluation with 10 elements/11 ordered nodes per axis including 0, 0.5, 1; original reference Ritz-energy error; offline learning followed by parameter-to-mesh prediction, with no prediction-time FEM/PDE solve or fresh instance-specific mesh optimization. These fix the physical problem and discretization size for comparison across predictors.

**Editable methods.** coefficient-to-mesh predictor architecture, internal representation, training surrogate loss and optimizer; training-time FEM/mesh optimization on the designated training data, distillation and equivalent geometry-preserving prediction implementations.

**Forbidden substitutions.** replacing the physical PDE, reference energy, finite-element order, domain or mesh-size problem; turning held-out parametric prediction into fresh per-instance FEM/mesh optimization, reconstructing withheld coefficient rows or using reference-energy lookup tables. When retaining the supplied training pipeline, changing only --lr=0.01, --lr-schedule=fixed or --steps=1400000 does not count as a method improvement.

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
