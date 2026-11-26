# Beyond Description: Federated Adaptation via Semantic-Visual Prototype Alignment

## 1. Goal

Improve `genacc` (higher is better) for UCF101 classification using frozen CLIP
ViT-B/16 embeddings. The baseline uses federated semantic–visual prototype alignment. The setting
uses an eight-shot labeled pool, ten clients, Dirichlet rho 0.1 and ten communication
rounds. Personalization accuracy must remain at least 86.5%. The judge retrains your
code with its own algorithm seeds and evaluates unseen inputs. You may change
the method within the constraints below.

## 2. Data

- `data/train/train.npz`: 646 training rows, with `id,x,y,client,rank`.
  `x` has 512 float32 frozen visual features; labels are class indices 0..100.
  Client indices are 0..9; rank preserves the original within-client sample order.
- `data/text.npz`: fixed normalized class text embeddings, shape 101 by 512.
- `data/val/input.npz` and `data/val/labels.csv`: 162 validation rows with public
  labels. Twenty percent of the original eight-shot pool was held out by stratified
  splitting with seed 20261030. Each class retains six or seven training rows.
- `data/val/refit.npz`: those validation rows with the original training
  augmentation and client assignments, retained for validation diagnostics only.
  Despite the historical filename, these rows must never be used for fitting.
- The hidden evaluation split is held by the judge. Its labels and client
  assignments are unavailable.
- The total labeled pool has exactly eight examples per class. The baseline fits
  only `train`; candidates must also fit only those training rows, including final
  models. The held-out validation rows are reserved for evaluation and selection. Do not acquire external data, change assignments,
  resplit, relabel, or infer hidden rows. IDs are identifiers, never model features.

## 3. Metric and scoring

`genacc` is the global top-1 accuracy in percent. PerAcc is the unweighted mean of
ten client accuracies using the corresponding personalized head. Output all ten
personal heads; the scorer selects the correct one using private client labels.

| Guard | Enforcement |
| --- | --- |
| PerAcc at least 86.5% | scorer |
| Frozen CLIP ViT-B/16 feature encoders | frozen data, certificate check, and patch audit |
| Eight-shot total pool; K=10; rho=0.1; ten communication rounds | materialized data, certificate probe, and patch audit of actual training |
| At most C × 512 uploaded floats per client per round, C=101 | certificate probe; patch audit of actual communication |
| Predictions agree with exported model state and depend only on their own input row | certificate check and patch audit |
| Fit on train only; validation for evaluation/selection; no hidden information | patch audit |

A violation has semantic worst score 0. The scorer emits `Infinity` as the harness
failure signal so invalid runs cannot be averaged into a successful result.
Structural rules require patch review. Run
`python bench/check_model.py MODEL --input data/val/input.npz --pred PRED`
to check the supplied model certificate and independently recompute its outputs.

The measured validation baseline is 80.04115226% GenAcc
(population standard deviation 0.29099044 percentage points), with
mean PerAcc 90.63340336%. `bench status` also reports this baseline. The judge uses three algorithm seeds. PASS requires a
relative mean improvement of at least max(0.012, twice the reference standard
deviation divided by its mean), while satisfying every guard.

## 4. Baseline

FedSPA, official source https://github.com/eejiarong/FedSPA-main at upstream commit
`3900107abc9b4a2b30837902f2bd1287a8996899`, is under `code/`.
`utils/client.py` implements visual adapters and `utils/server.py` implements
prototype aggregation and regularized InfoNCE semantic updates.

The baseline uses frozen cached features, separates training and prediction,
saves complete model state and a certificate, and reads seeds/threads from the
benchmark environment. It always returns the final round.

Runtime at 16 CPUs: mean training 2.391 minutes, mean validation
prediction 0.067 minutes. Federation uses up to four torch
threads. The 16 GB memory allowance is not a measured peak.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` trains and saves everything prediction needs.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with
  one global and ten personal class predictions per row. See `bench/io_spec.md`.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, and `BENCH_BENCH_DIR`.
- Use `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`,
  and `bench submit --final -m MESSAGE` for the development workflow.
- Dependencies are pinned in `code/env.lock`. Install with `uv pip install`;
  submission freezes installed packages. Only reproducible `name==version` pins
  from PyPI or the PyTorch CPU index are supported.

## 6. Rules

- You may replace the learned adaptation/head method, training surrogate and optimizer under `code/`, subject to the
  metric guards, data budget and the fixed entry points and prediction format.
- `data/`, `bench/`, this README, and the task specification are frozen.
- Resource allocation: 16 CPUs and 16 GB memory.
- Every retraining run must honor BENCH_SEED. Seed variance counts against you.
- Reading or guessing hidden rows, hard-coding predictions, memorizing reported
  results, changing client assignments, and falsifying the certificate are
  disqualifying. Self-reported training metadata does not replace patch review.
- Preserve the eight-shot total data pool, ten clients, rho 0.1, ten rounds,
  the supplied frozen feature inputs, and communication budget even when changing the method.

## 7. Known pitfalls

PerAcc is a client mean, not ordinary sample accuracy. Personalized heads use
semantic prototypes from before the final server update, whereas the global head
uses the final update. Do not expose client membership to prediction code.
Training embeddings retain the original random augmentation; validation uses
deterministic evaluation preprocessing. Fixed cached features are never re-encoded
or fine-tuned. No image or text encoder download is needed at runtime.

Output paths may have no extension. Write exact paths and sorted, unique input
IDs. The scripts set PYTHONPATH explicitly for PYTHONSAFEPATH mode. Only committed
files reach the judge's clean clone. The model uses safe NPZ and JSON, not pickle.
The certificate check verifies prediction consistency. Actual training rounds
and communication require patch review.

## Research scope

Objective: Improve federated UCF101 generalization on frozen CLIP ViT-B/16 features while retaining personalized accuracy.

Fixed research objects and target: The supplied frozen image/text embeddings, eight-shot total labeled pool, ten clients with fixed rho=0.1 assignment, ten communication rounds and per-client-per-round upload limit 101*512 floats remain fixed. Global accuracy is the objective; the unweighted ten-client personalized accuracy must be at least 86.5%.

Editable methods: The learned heads, prototype/adaptation method, training surrogate, local optimizer and aggregation strategy may change within the fixed feature, federation and communication setting.

Forbidden substitutions: Do not change or fine-tune the CLIP encoders, replace the supplied features, centralize private client inputs, change client assignments/rounds/uploads, expose private client membership at prediction, or alter either accuracy rule. When retaining the supplied FedSPA pipeline, changing only alpha=9, beta=8, local_lr=0.0004, global_lr=0.0004, lam_reg=0.2, temperature tau=0.01, local_batch_size=8, global_batch_size=8, local_epochs=5, local_epochs_last=100 or local_epochs_server=100 does not count as a method improvement.

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
