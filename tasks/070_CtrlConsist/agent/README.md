# Control Consistency Losses for Diffusion Bridges

## 1. Goal
Learn a reusable controller for the CIR diffusion bridge dX_t = a(b-X_t)dt + epsilon*sqrt(X_t)dW_t, with initial and terminal state 2, horizon 1, and a=b=epsilon=1. This tests learning a bridge with state-dependent diffusion. Improve `kl_to_solution` (lower is better), the fixed Monte Carlo KL(P* || P_theta), where P* is the endpoint-conditioned CIR law and P_theta is the learned controlled process. The baseline regresses control values onto targets computed backwards along its own simulated trajectories. You may change the learned method subject to the interface and observation rules below.

## 2. Data
- `data/train/config.yaml` contains the released CIR problem and baseline settings. There is no fixed training dataset: training simulates its own controlled trajectories using the base drift and diffusion.
- `data/val/input.npz` and `data/val/labels.npz` contain 5000 public validation trajectories in five independent Monte Carlo groups, generated with public seeds 20261001 through 20261005. Inputs are opaque trajectory IDs, observed states, times and controller repeat slots; labels contain true drift, sigma and evaluation metadata. Use labels only for offline validation and configuration selection.
- No evaluation random keys are supplied to prediction.
- `bench/generator.py` freezes the released float32 CIR sampling process. Do not use its true-drift oracle or validation labels to train a supervised substitute or compute predictions. Training may simulate the base and learned controlled SDEs. Do not relabel, re-split or import outside data.

## 3. Metric and scoring
`bench/scorer.py` computes the released `endpoint_idx=-1` discretization: per path, one half of the sum of squared true-minus-predicted total drift divided by sigma squared, multiplied by dt=1/500. It averages paths within each controller repeat, then averages the five independent repeats. It does not average the controllers before scoring. Float32 arithmetic and the original reduction order are retained.

| Guard or rule | Enforcement |
| --- | --- |
| Exact output keys, unique matching IDs, correct tensor shape and finite drift values | scorer and validator |
| Predictions agree under path subsetting and row permutation | patch audit; optional prediction consistency check |
| Each value uses only its own (x,t) and its fitted repeat controller, with no future-state or cross-path features | patch audit; probe detects some violations |
| No analytic true-bridge drift, label lookup, hard-coded answers or extra ground-truth simulation in training/prediction | patch audit |

The score rule has no additional statistical constraint. Malformed outputs give Infinity. An exact drift tensor mathematically scores zero; the origin of such predictions cannot be determined from the tensor alone, so oracle misuse requires patch review. The judge runs three independent groups of five training seeds. PASS uses relative improvement at least `max(delta, 2*reference_std/reference_mean)`, with delta=0.05. Measured validation baseline: **0.020877622192**, population standard deviation across the three groups **0.002380534730**. These are independent groups of five trained controllers, not individual-model standard deviations.

## 4. Baseline
Official implementation: https://github.com/samuel-howard/control_consistency_bridges, upstream commit `1ec101dc6f8d1d4c1e1a0494fedd6d7fa6484e16`. `src/models.py` defines the conservative MLP, `src/compute_targets.py` the SC1 targets, and `src/training.py` the loss and update loop.

Baseline settings remain the authors': hidden layers [128,128], time embedding [64,64], GELU, 2000 outer iterations, 500 SDE steps, 64 sampled paths, 20000 training pairs per iteration, Adam learning rate 1e-4, original clipping and EMA, final saved EMA checkpoint. Five independent processes share the 16 CPUs in disjoint groups. The memory budget is 32 GB, not a measured peak. Measured runtime at 16 CPUs: complete five-controller training 15.23-17.41 minutes; validation prediction 0.12-0.16 minutes.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` trains five controllers and writes all artifacts into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes NPZ arrays `id` and `drift`; see `bench/io_spec.md` for the tensor specification.
- Honor `BENCH_SEED` and `BENCH_CPUS`. Paths are supplied through `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`. For seed s, baseline training seeds are `5*(s-1)+i`, i=0..4. The same seed must be used for training and prediction.
- Use `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and `bench submit --final` for development.
- Dependencies are rebuilt using exact PyPI `name==version` pins in `code/env.lock`, Python 3.11, CPU JAX. `uv pip install` updates the development venv; submission freezes its packages. Editable, git and local-path dependencies are unsupported.

## 6. Rules
Controller parameterizations, auxiliary training losses, optimizers, numerical settings and fitting methods under `code/` may change while solving the same fixed CIR bridge problem, subject to the Forbidden substitutions below. The base SDE, endpoint conditioning and KL target cannot change. `data/`, `bench/`, this README, script signatures and output schema are frozen. Preserve five independent training repetitions; do not pool their predictions into an ensemble. Fit reusable controllers before prediction; prediction must not refit on supplied trajectories.

Do not use or reimplement the analytic true-bridge solution to generate training targets or predictions, infer evaluation seeds or identities, read labels during training/prediction, exploit future states or other paths, or embed remembered outputs. Closed-form truth is supplied only for offline validation. These semantic restrictions require patch audit. No network is available during train or predict. CPU budget is 16; memory budget is 32 GB.

## 7. Known pitfalls
The predicted quantity is total drift, including the guiding drift and sigma-scaled neural adjustment. The final time is excluded using the released -1 slicing; replacing it by -5 changes the metric. The baseline uses the last saved EMA checkpoint; alternative fitting methods may not fit on validation data, while checkpoint selection may use it. Write NPZ via a binary file handle, because the output filename has no guaranteed suffix. IDs may be shuffled and must be matched explicitly. JAX thread budgets matter when running five processes; XLA takes its thread count from `PJRT_NPROC`, not from `OMP_NUM_THREADS`. A scalar claimed KL is not a prediction. The fixed CIR parameters are public.

## Research contract

**Objective:** Learn the control/total drift of the fixed CIR diffusion bridge, minimizing the original KL(P*||P_theta) discretization against its true bridge.

**Fixed research objects:** The CIR base drift/diffusion with a=b=epsilon=1, initial/terminal state 2, horizon 1, and conditional bridge law P* remain fixed. Pointwise total drift at each supplied (x,t), five independent fitted repeat controllers, endpoint_idx=-1, float32 reduction and original Monte Carlo KL metric remain fixed.

**Editable methods:** Controller architecture, auxiliary learning loss, self-consistency/other valid control-learning algorithm, optimizer, simulation/discretization for training and checkpoint selection may change. SC1 targets, conservative MLP, EMA and training counts are baseline methods; permitted training simulation uses the same base/controlled SDE rather than a different bridge.

**Forbidden substitutions:** Do not change the SDE/diffusion parameters, endpoint/horizon conditioning or metric, use analytic true bridge drift as supervision/prediction, or fit using evaluation labels. No future states/cross-path features, refitting during prediction, combining five controllers into an ensemble, hidden seeds or added ground-truth simulations. When retaining the supplied SC1 target pipeline, changing only train.STL_adjustments=false does not count as a method improvement. Changing only train.num_outer_iterations=2000, traj_batch_size=64, lr=1e-4 or beta_schedule=average also does not count.

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
