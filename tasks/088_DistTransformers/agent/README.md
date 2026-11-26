# Distribution Transformers: approximate Bayesian posterior inference

## 1. Goal
Given supplied inverse-gamma prior parameters and one zero-mean normal observation, predict a two-component Gaussian mixture for the log variance posterior. This is the two-component, wide-meta-prior setting. Improve `posterior_kl` (lower is better) on independent problems with the same distribution as validation. The judge retrains your code from scratch.

## 2. Data
- `data/train/config.json` specifies the original synthetic training distribution and settings. Training generates 20 epochs of 100 batches of 5000 fresh problems; no external dataset is required. You may generate training instances from this distribution.
- `data/val/input.npz` and `data/val/truth.npz` contain 1000 validation problems generated with public seed 20261029; labels are for evaluation and method, hyperparameter or checkpoint selection only.
- Hidden evaluation data are unavailable in this workspace.
- `bench/generator.py` freezes the data-generating process. Input contains opaque IDs, supplied prior concentration/rate, and the observed response. The unknown variance and scoring randomness are labels. Meta-prior concentration and rate are independently InverseGamma(4,6). For supplied concentration c and rate r, the variance x follows InverseGamma(c,r), and z given x is Normal(0, variance=x). The posterior is InverseGamma(c + 0.5, r + z^2/2).
- Do not alter or re-split the frozen files or use external data. Extra simulation is allowed for training, as in the original method; prediction applies the same fitted amortised mapping to each supplied prior/observation.

## 3. Metric and scoring
`posterior_kl` is the mean Monte Carlo KL from the exact inverse-gamma posterior to the predicted log-Gaussian mixture, using 10000 samples per problem. For a posterior sample x, exact posterior density p and predicted log-variance density q, the contribution is log p(x) - log q(log x) + log x. The last term is the log-transform Jacobian correction. Scoring uses fixed random samples, restores original draw order, and averages draw means. The theoretical lower bound is zero; individual finite Monte Carlo estimates can be negative and are not clipped.

| Guard | Enforcement class |
|---|---|
| Exactly two valid Gaussian components with positive weights and scales | scorer; judge model replay probe |
| Amortised inference with no per-problem fitting or posterior-sampling chain | runner declaration/immutability checks and patch audit |
| Independent per-problem inference from fitted state and that problem's supplied prior/observation | judge state/subset replay probe; patch audit |
| No hidden labels, seed recovery, memorised answers, or bypass of the frozen runner | patch audit |

Violation gives `Infinity`. Forward timings are diagnostics measured by the frozen runner; model loading, I/O and scoring are excluded. An unmodified, randomly initialized DT supplies a paired diagnostic timing reference.

Baseline validation `posterior_kl` = **0.0057068324**, with training-seed population standard deviation **0.0000120511**. Three independent training seeds are used. PASS requires relative improvement at least `max(delta, 2*reference_std/abs(reference_mean))`; delta is 3%. Model replay is an operator probe: CSV claims alone cannot establish input provenance or independent inference, and the operator also audits the patch.

## 4. Baseline
Official code: https://github.com/GWhittle110/distribution-transformers, reported upstream revision `8c28f969e350581838b1db176dfa0d0251544eb4`. The supplied source is the baseline snapshot.

The model and loss live in `code/src/model/` and `code/workflows/train.py`: 2 components, 6 decoder layers, width 64, 8 heads, feedforward width 256, dropout 0, AdamW learning rate 0.005, weight decay 0, 5 warmup epochs, 20 epochs, 100 steps per epoch, batch size 5000, prior plus posterior loss. The final checkpoint is used without selection. The original parameter gradient clamp is retained.

The baseline uses separate offline train and predict entry points. On 8 CPUs, reference training takes 308.11 to 312.56 seconds. Validation prediction takes 4.62 to 4.72 seconds; scoring is a separate stage. The memory budget is 16 GB; peak RSS was not measured.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes all fitted state.
- `bash code/predict.sh --model DIR --input PATH --output FILE` emits the CSV specified in `bench/io_spec.md` through the frozen prediction runner.
- `model_api.load_model(model_dir)` and `model_api.forward(model, phi, z)` expose fitted state and return `(n,2,3)` NumPy or Torch mixture parameters. Forward uses only fitted state and each row’s inputs.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, and `BENCH_CODE_DIR`.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and `bench submit --final` are the development/submission interfaces.
- Dependencies are rebuilt from `code/env.lock`, with only exact version pins from PyPI or the PyTorch CPU index.

## 6. Rules
You may change the model, features, losses, optimizer and training procedure under `code/`, and retune hyperparameters as part of such a change, subject to the two-component output, independent per-problem inputs and fixed interfaces. The `data/`, `bench/`, README and I/O specification are frozen. Keep the entry point invoking the frozen runner without interception, clock manipulation, altering baseline timing, monkey-patching trusted modules, or fabricating CSV fields.

Every prediction is a function of its own supplied prior/observation and fitted state. Apply a shared fitted inference mapping; do not fit a fresh approximation, run a posterior-sampling chain or update fitted parameters for the evaluation problem. Scoring samples, hidden posterior state and labels are unavailable. No reading labels, truth tables, ID-dependent predictions or memorised results. Reading or attempting to recover hidden seeds/data is disqualifying. No network access during training or prediction. Training seeds are controlled by the judge; variation counts against you. The CPU budget is 8 and memory limit 16 GB.

## 7. Known pitfalls
The two Gaussians model **log variance**. Output standard deviations, not variances; forgetting the Jacobian changes KL. Reordering input is allowed, but IDs must be preserved exactly. Use file handles if writing NumPy archives to extensionless paths. The scripts set PYTHONPATH because the sandbox uses PYTHONSAFEPATH=1. Predictions must agree under replay and row permutation and must not mutate fitted state. Model evaluation retains the original training mode with zero dropout; changing mode can select a different numerical attention implementation. Public validation labels are for evaluation and method, hyperparameter or checkpoint selection only, never for fitting or inference. The full training schedule is substantial; budget development runs accordingly.

## Research scope

**Objective.** Improve prior-flexible amortised two-component approximation to the specified inverse-gamma posterior.

**Fixed research objects.** The supplied prior/observation distribution, normal likelihood, latent variance and log-variance output space are fixed. Inference is amortised across problems; output is exactly two valid Gaussians. The original posterior KL Monte Carlo formula, Jacobian and aggregation remain fixed.

**Editable methods.** The shared inference architecture, fitted representation, offline training method, training surrogate and optimizer may change; a Transformer is not mandatory.

**Forbidden substitutions.** Do not replace the Bayesian problem, fit each evaluation problem separately, run a posterior-sampling chain, expand the mixture family, change KL scoring, access labels or transfer information across evaluation problems. When retaining the supplied DT training pipeline, changing only epochs=20, steps_per_epoch=100, batch_size=5000, warmup_epochs=5, lr=0.005, weight_decay=0 or use_prior_loss=true does not count as a method improvement. Changing only the width or depth in `transformer_kwargs` (num_layers=6, d_model=64, n_head=8, dim_feedforward=256) also does not count.

The operator replay checks output consistency and fitted-state immutability. The declaration does not prove amortisation; actual source execution and data provenance require audit.

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
