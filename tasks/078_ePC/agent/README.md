# ePC: predictive-coding classification on FashionMNIST

## 1. Goal
Improve `accuracy` (higher is better) by improving predictive-coding learning of the specified FashionMNIST four-layer MLP with a cross-entropy output objective. Predict the garment class from a grayscale image. Each evaluation trains five independent replicas and averages their individual accuracies, using the fixed predictor and a lawful predictive-coding learning method. Hidden evaluation data are held by the judge.

## 2. Data

- `data/train/train.npz`: 48,000 images, with `id`, uint8 `x` of shape N × 28 × 28, and integer class `y` in 0–9.
- `data/val/input.npz` and `data/val/labels.csv`: 12,000 validation images and public labels. This is a stratified 20% holdout from the original training data, split seed 20261030. Each file is sorted by its original row id.
- Hidden evaluation data are not available to candidates.
- The baseline trains on `train` only. Model selection may use validation. Final candidate models must also fit only on train. Use only the supplied images and labels; no external data, reconstructed held-out data, relabeling, or replacement splits.
- Baseline preprocessing: float32 `/255`, subtract 0.5, divide by 0.5, flatten in row-major order. This canonical input representation and the supplied image files/labels are fixed; equivalent computation is allowed.

## 3. Metric and scoring
`accuracy` is `100 × mean_s(mean_i(pred_s[i] == y[i]))`, with replica slots s=0–4. This is the mean of five accuracies, not ensemble accuracy. `bench/scorer.py` also reports the five individual values and integer correct counts. Invalid submissions have rule score zero; detected violations emit `Infinity` as the harness failure sentinel, which always fails the submission.

| Guard | Enforcement |
|---|---|
| Five class columns, finite integer classes 0–9, exact ordered id set, no duplicates | scorer and validator |
| Fixed-MLP tensors, manifest, seed mapping and reconstructed classes | judge probe `bench/check_model.py`; PC training provenance still requires patch audit |
| Five independently trained replicas, honest seed handling and declarations | patch audit |
| Training and selection use only allowed data | patch audit |

`train.sh` must write `model_card.json` and all prediction artifacts described in `bench/io_spec.md`. Run `python bench/check_model.py MODEL_DIR --input data/val/input.npz --pred PRED_FILE` yourself. It reconstructs the fixed MLP from tensor-only artifacts without importing candidate code. The probe does not run automatically. Actual PC training, lawful data use and replica independence require patch audit.

Baseline validation accuracy is **88.240000%**, with group standard deviation **0.055428 percentage points** (three groups of five replicas). The judge runs three groups and compares their mean against its archived baseline. PASS requires relative improvement of at least `max(delta, 2 × baseline group standard deviation / baseline mean)`, with delta=0.003. See `bench status` for the measured reference.

## 4. Baseline
Official code: https://github.com/cgoemaere/error_based_PC at upstream commit `95c555197699f6b4d15452451da15049946b49ca`. The predictive-coding implementation is in `pc_e.py`, `get_arch.py` and `pc_variants.py`.

The baseline uses orthogonal ReLU-gain initialization and zero biases, three GELU hidden layers and a linear output. It uses cross-entropy with one-hot targets, four error-inference SGD iterations at 0.003, local weight updates through Adam at 5e-5, batch size 64 with shuffled training batches and `drop_last=True`, and 14 epochs. It saves the final epoch without early stopping. Error inference holds weights fixed and minimizes squared hidden errors plus the classification loss. Local weight learning holds inferred states fixed through detachment.

Each group contains replica slots 0–4. Actual RNG seeds are `(5*(BENCH_SEED-1)+slot) mod 2**32`, so group seed 1 uses 0–4. Training uses up to five independent processes, each with at most two threads, within the 16-CPU budget. Dependencies preserve the source versions, including CPU torch 2.1.1 and Lightning 2.3.0. Measured runtime on 16 CPUs: training a complete five-replica group takes 1.54–1.57 min (mean 1.56); validation prediction takes 6.32–6.37 s. Memory limit is 16 GB; peak resident memory was not measured.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` trains five models and saves all prediction artifacts to DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` predicts every input row. Exact format: `id,pred0,pred1,pred2,pred3,pred4`; see `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`. Scripts resolve arguments before changing directory and set their own import path.
- Development commands: `bench status`, `bench eval-dev --seed S`, `bench validate-output FILE`, `bench submit --final`.
- Install dependencies into the active venv; only `name==version` pins from PyPI / PyTorch CPU indexes are rebuildable in `code/env.lock`.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Improve the PC inference solver and local weight updates while keeping the specified MLP, GELU activations, CE output objective and canonical input representation. Optimizers, learning rates, inner iteration policies, initialization, batch scheduling and training duration may change as part of a changed PC method. Direct end-to-end backpropagation is outside this research problem. Reverse-mode autodiff may implement PC computations, but it must not replace PC inference/local learning or intentionally collapse to the exact-BP limit. For error-based PC, a single error update produces a BP gradient scaled by the error learning rate. As the product of error step size and iteration count tends to zero, the local update approaches a scaled BP gradient. Model artifacts and declarations must describe the actual submitted method.

`data/`, `bench/`, `README.md`, prediction format and entrypoint signatures are frozen. The judge retrains from scratch in a clean clone with no network. Files needed at runtime must be committed; pretrained weights are not allowed. Honor BENCH_SEED for every group. Final models fit on train only; validation is available for evaluation and selection only.

Resource allocation: 16 CPUs and 16 GB RAM. Reading or reconstructing the hidden split, hard-coding predictions, memorizing paper results, and submitting misleading certificates are disqualifying.

## 7. Known pitfalls
Training means five complete runs per call; a single replica or ensemble vote violates the metric. Keep `PYTHONPATH` explicit because the harness sets `PYTHONSAFEPATH=1`. All data mounts are read-only; write artifacts only to output or temporary directories. Predict all supplied rows, including incomplete final batches. Compare against the benchmark validation reference, which uses the supplied training split. The fixed model probe reconstructs classes from the saved weights. Final tensors cannot prove that PC training, CE, lawful data use and replica independence were respected; those require patch review.

## Research scope

**Objective.** Improve predictive-coding learning of the specified four-layer FashionMNIST MLP with a cross-entropy classification objective, scored by mean accuracy of five independent replicas. Keeping the predictor and output objective fixed makes improvements comparable as changes to PC learning.

**Fixed research objects.** 784→128→128→128→10 feed-forward MLP, three GELU hidden activations and a linear output, 134794 parameters; predictive-coding inference and local weight-learning paradigm with cross-entropy output objective; canonical float32 image normalization and flattening, supplied data/splits, five independent replicas and seed mapping.

**Editable methods.** local error/state inference solvers, convergence criteria and nondegenerate PC variants; local weight-update optimization and, within a changed method, learning rates, initialization, batch scheduling and training duration; lawful development selection; equivalent implementations of the fixed predictor and runtime efficiency.

**Forbidden substitutions.** replacing the specified network with a CNN, a different architecture or feature map, or exceeding its parameter budget; replacing predictive-coding learning with direct end-to-end backpropagation, including a disguised exact-BP limiting construction; changing the CE research cell to MSE/another output objective or replacing five-member accuracy with ensemble accuracy. When retaining the supplied ePC pipeline, changing only iters=4, e_lr=0.003, w_lr=5e-5, batch_size=64 or max_epochs=14 does not count as a method improvement.

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
