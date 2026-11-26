# {{title}}

Task statement for the agent. Everything below is binding: the judge reruns your `code/` from the prescribed
starting state with its own seeds, scores the hidden test split, and compares against the archived baseline.
<!-- template note (delete): keep only the bullets that apply to this task's type; never state the exact size of the
test split or any test file name; refer to the test split only as "held by the judge". -->

## 1. Goal
<One paragraph, in the paper's own terms: what is predicted or decided, from what input, under which protocol
(the paper's standard experimental setting). Then the objective: improve `{{metric}}` ({{direction}}) on the
hidden test split, which has the same distribution and preprocessing as `data/val/`. Identify the fixed research
objects and optimization objective, and the methods that may be changed without changing the problem; see §6.>

## 2. Data
- `data/train/...` — training split: <files, format, size, columns / units>
- `data/val/input.csv` + `data/val/labels.csv` — validation split (your dev set; labels are public)
- test split — held by the judge, <how it was formed: the paper's standard split / carved from X by rule Y / hidden seeds>; it is never readable from this workspace
- <synthetic tasks: `bench/generator.py` is the data-generating process; val = the public seeds in `data/val/`, test = hidden seeds. The input of `predict.sh` is observed data only; the true parameters are the labels.>
- <online tasks: `bench/env.py` is the environment; your algorithm may call only its public interface. Reading hidden state is a protocol violation and is checked.>
- Do not re-split, relabel, or augment the data from outside sources unless this section allows it.

## 3. Metric and scoring
- `{{metric}}` ({{direction}}), computed by `bench/scorer.py` exactly as <paper table / equation>.
- Guards (the paper's protocol constraints; violating any one sets the score to the worst value):
  - <guard 1 — enforced by: scorer | judge probe | patch audit>
  - <guard 2 — …>
- Baseline on val: `bench status` prints it (also `baseline.val` in the task). The judge scores the test split over k seeds and compares the mean with the archived baseline reference; PASS = relative improvement ≥ delta (also in `bench status`).

## 4. Baseline
- Method: <name>, official code <repo URL> at commit <sha>, under `code/`; <where the model / loss / training loop live>.
- Our protocol-only adaptation (`git log` in `code/`, commit "bench-adapt"): <data entry via BENCH_DATA_DIR; model selection on val only; save/predict entry points; seeds and threads from env; removed absolute paths>. Model, loss, hyperparameters and training loop are the authors'.
- Baseline runtime on {{cpus}} CPUs: train ≈ <X> min, predict ≈ <Y> min, memory ≈ <Z> GB.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` — trains and writes everything `predict.sh` needs into DIR
- `bash code/predict.sh --model DIR --input PATH --output FILE` — writes the prediction file; format in `bench/io_spec.md`
- env: `BENCH_SEED` (int, must be honored), `BENCH_CPUS` (thread budget), `BENCH_DATA_DIR` (= `data/`), `BENCH_BENCH_DIR` (= `bench/`)
- `bench status` · `bench eval-dev [--seed S]` (train → predict on val → validate → score, exactly the judge's pipeline) · `bench validate-output FILE` · `bench submit [--final] [-m MSG]`
- Dependencies: `uv pip install …` into the venv; `bench submit` freezes it into `code/env.lock`. Only `name==version` pins from PyPI / PyTorch-CPU are rebuildable; editable, git or local installs fail the build.

## 6. Rules
- Fixed research objects and objective: <name the specified model/checkpoint/architecture, physical or statistical system,
  target function/distribution, and experimental controls that define this problem; do not infer these merely from the baseline>.
- Editable methods under `code/`: <list the task-specific algorithms, model components, training surrogate losses,
  optimization procedures or implementation choices that may change, including complete method replacement where valid>.
- Forbidden substitutions: <concrete changes that would replace the research object or optimization target; for example,
  substituting a smaller unrelated model for the specified model in a pruning task>.
- A model, loss or architecture may be changed only if it belongs to the editable method, rather than the fixed research object or objective.
- Frozen surface: `data/`, `bench/`, `README.md`, the prediction format, the script signatures. A patch touching them is invalid.
- Resources: {{cpus}} CPUs, memory ≤ <M> GB. No network during train/predict: anything needed must be in the patch or installable from `env.lock`.
- Submit a valid patch with `bench submit` before the overall deadline; no valid submission means failure.
- Seeds: the judge runs k seeds; every run must honor `BENCH_SEED`. Variance across seeds counts against you.
- Disqualifying: reading or guessing the test split, hard-coding predictions, embedding results memorized from the paper, bypassing the environment/generator interface.

## 7. Known pitfalls
<memory, data quirks, numerical conventions, thread settings, and task constraints that are easy to miss>

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
