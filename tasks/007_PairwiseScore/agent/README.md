# Score-Based Density Estimation from Pairwise Comparisons

## 1. Goal

An expert holds a belief density p(x) that cannot be observed. We only see noisy pairwise comparisons: two candidates are drawn independently from a known proposal and the expert picks a winner. The expert follows a Bradley-Terry model with utility log p(x) and Gumbel(0, s) noise, s = 0.7797 (unit variance). Research question: how accurately can p(x) be recovered from such comparisons alone?

The experiment follows the eight synthetic Table-1 targets of Mikkola et al., ICLR 2026: Onemoon2D, Twomoons2D, Ring2D, Gaussian4D, Mixturegaussians4D, Stargaussian6D, Mixturegaussians10D and Gaussian16D (groups 0 to 7 in the labels). The paper compares a normalizing flow, score-tau* (constant tempering) and score-tau(x) (position-dependent tempering field) on the same targets, comparison counts and metrics. The estimator is what varies. Score-tau(x) is the baseline here; other estimators may answer the same question.

Fit each instance independently and return samples in original coordinates. Improve `mmtv` (lower is better) while satisfying the Wasserstein guards. Evaluation instances held by the judge come from the same generators and preprocessing as the public train and validation instances.

## 2. Data

- `data/train/input.npz` and `data/train/labels.npz`: eight public training instances, one per target, observation seed 100.
- `data/val/input.npz` and `data/val/labels.npz`: eight public validation instances, one per target, observation seed 101.
- Each instance holds 1000·D comparisons in winner/loser order. Proposals are uniform on a box for D=2 and D=4, and known Gaussian mixtures for D=6, 10 and 16. D=2 comparisons stay in original coordinates. For larger D they are already mapped to [-0.5, 0.5]^D, linearly for D=4 and by a Rosenblatt transform of the proposal above that, so samples must be mapped back.
- `bench/generator.py` and its frozen dependencies specify the released generation process. Hidden instances use independently secret seeds and are held by the judge.
- Inputs contain only ranked comparisons, opaque shuffled IDs, and the known proposal/calibration metadata. Labels contain evaluation target samples and grouping information; use them for offline diagnostics and configuration selection only.
- The 2D baseline preconditioning scale uses the original preliminary 10,000-point calibration sample's empirical standard deviation. Only that scalar is provided. See `bench/io_spec.md` for details.
- Each instance is fitted on its own comparisons inside predict. There is no shared training step and no cross-instance learning, and the baseline learns nothing from `data/train`. The public instances serve evaluation and method, hyperparameter or checkpoint selection only.
- No re-splitting, relabeling, external data, extra target simulation, target-table lookup or inference of generator seeds is allowed.

## 3. Metric and scoring

`mmtv` averages marginal total variation over coordinates, replications within each target, and targets. The frozen scorer uses the released KDE and one-million-point numerical integration. Wasserstein uses the original Euclidean transport cost and POT's default iteration cap on the leading 15,000 rows (D=2, 4) or 20,000 rows (D=6, 10, 16).

| Guard | Requirement | Enforced by |
|---|---|---|
| Output format | Exact IDs, full row counts, D columns, finite floats | validator |
| Average relative Wasserstein | Mean over targets of W / frozen baseline W ≤ 1.05 | scorer |
| Worst relative Wasserstein | Maximum target W / frozen baseline W ≤ 1.40 | scorer |
| Empirical sample validity | Samples cannot collapse to one point or exactly copy the evaluation target marginals | scorer |
| Observations only | Use only an instance's supplied observations/metadata and a fixed configuration selected on the public instances; no target lookup, labels or extra target simulation | patch audit |
| Subset consistency | Predictions agree when an instance is run alone | judge probe |
| Independent instances | No information from other evaluation instances or their order | patch audit |

The Wasserstein baseline is measured on the same instance collection and frozen in its labels. Violating a Wasserstein guard sets the metric to **1.0**. Invalid or nonfinite tensors fail validation. Collapsed sample distributions and exact copies of evaluation target marginals receive Infinity as protocol violations. Exact-copy detection misses other truth-table samplers, and a perfect score cannot establish whether target-table lookup occurred: that rule rests on patch audit. A separate subset-replay probe can check instance independence outside the automatic scoring.

Public validation MMTV is **0.21864791 ± 0.01736092** (mean ± population standard deviation over three algorithm seeds). The judge uses three algorithm seeds. PASS requires improvement at least max(5%, twice the reference seed standard deviation divided by its mean). Both reference statistics come from the baseline's run on the hidden instances, not from the validation numbers above.

## 4. Baseline

Official code: https://github.com/petrus-mikkola/pairwise2diffusion at commit `590aaac66f38db67e99b788e69ee55a5ee765039`. Baseline is score-tau(x): the released score network, Adam training, learned tempering field and annealed Langevin sampler. `model.py`, `diffusion.py`, `tempfield.py`, and `likelihood.py` implement the method; `cpu_main.py` retains its training loops. CPU kernels and transform caches (`cpu_tau.cpp`, `cpu_tempfield_opt.py`, `cpu_transformation_opt.py`) speed up the released computation.

The key result: on a uniform proposal, the score of p(x) equals tau(x) times the score of the marginal winner density p_w(x), and tau(x) has a closed form in the density ratio p(x')/p(x) and s. Per instance the baseline (1) trains a score network on winner/loser pairs by denoising score matching, masking the loser half of the time to get the p_w score; (2) trains a density-ratio network on the Bradley-Terry likelihood and estimates tau(x) by importance sampling with 2000·D winner-model draws, clipped between 1 and its 0.99 quantile; (3) samples by annealed Langevin dynamics with the score scaled by tau(x). The paper reports that this setup is sensitive to hyperparameters and that overestimating tau(x) can give too-tight marginals in higher dimensions.

Comparisons come frozen in the input files, with separate observation and algorithm random streams. Train writes configuration; predict fits each instance. Model, loss, hyperparameters, iterations and per-instance sample counts are the released ones. Only the scored varying-tempering branch is run. At 128 CPUs, train takes 0.47-0.52 seconds and public validation prediction takes 21.34-25.22 minutes. The configured memory budget is 320 GB.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` compiles the shipped CPU kernel and saves configuration.
- `bash code/predict.sh --model DIR --input PATH --output FILE` fits each supplied instance independently and writes NPZ sample tensors.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Honor the seed and CPU budget. The original baseline derives independent per-instance algorithm seeds from BENCH_SEED and opaque ID.
- `bench eval-dev --seed 1` runs one seed. `bench validate-output FILE` checks a prediction file.
- PyPI and PyTorch CPU wheels are supported; editable, git or local installs are not.

## 6. Rules

- Editable surface: density estimators, representations, training surrogate losses, optimizers and samplers under `code/`, within the fixed comparisons-to-density research scope below.
- Frozen surface: `data/`, `bench/`, this README, output format and script signatures.
- The judge uses 128 CPUs and a 320 GB memory budget. There is no network during train/predict.
- Every run must honor BENCH_SEED. Public labels are for offline evaluation/configuration selection; do not fit a truth-table predictor or use known density formulas from the published paper or the generator to synthesize outputs (patch audit).
- Reading or guessing hidden data, hard-coding predictions, recovering generator seeds, using target identity inferred from dimension or proposal metadata to retrieve a density, and accessing another instance's observations are disqualifying protocol violations requiring patch audit.

## 7. Known pitfalls

The method trains two networks per instance; the short train entry point only prepares configuration. A prediction call is expensive, so start with a public subset when debugging. The baseline keeps the released thread layout: 4 threads for each 10D/16D worker and 2 for each smaller worker; the sum never exceeds BENCH_CPUS.

Public validation has one instance per target, so it is noisy. Its seed standard deviation is about 7.9% of the mean, and Mixturegaussians10D alone ranges from 0.12 to 0.38 across the three seeds. A single-seed validation gain is weak evidence.

The original ring generator's batch-shared radius bug and capped OT convergence warnings are retained. These affect metric variability. Each ring evaluation target lies on one circle whose radius is drawn once, while the comparisons follow the ring density, so exact density recovery does not reach zero MMTV on that target.

Output must be actual full-count finite sample matrices, in original units; self-reported metrics are rejected. NPZ writing through a filename can append an unwanted extension: open the requested output path as a binary file handle. Scripts receive absolute paths and run with PYTHONSAFEPATH=1; preserve explicit package-path setup. The frozen scorer and validator use their own dependency environment.

## Research scope

**Objective.** Estimate each original multivariate belief density from its noisy ranked comparisons and return full samples in original coordinates, minimizing MMTV under the Wasserstein guards.

**Fixed research objects.** Original ranked-comparison observations and known proposal/calibration metadata for the eight target distributions; independent observation-only instance fitting, original-coordinate sample interface and dimension-dependent full sample counts; original marginal-TV/KDE integration, leading-sample Wasserstein computation and relative mean/worst guards. These stay fixed so that estimators remain comparable.

**Editable methods.** Density/score estimators, neural or other representations, training surrogate losses and optimization; the fitted tempering field tau(x), numerical samplers and internal transformations preserving the same inferred density and output units.

**Forbidden substitutions.** Use proposal means, proposal covariances and `sigma_data` only as the baseline does, to map between original and unit-cube coordinates and to precondition the score model; do not use them to set the location, scale or component positions of the estimate (patch audit). Apply one estimator family to every instance; numeric hyperparameters and the released preprocessing may depend on dimension, but no choice may depend on which of the eight targets an instance appears to be (patch audit). The sampler keeps the fitted input-dependent tempering field tau(x): do not replace it by a constant such as its mean or quantile, the paper's score-tau* variant, on any instance, including by forcing tau(x) to be constant; a gain must come from a better tau(x), score model or sampler (patch audit). Using published/generator target-density formulas or target identities as lookup answers, extra population-target simulations or another instance's observations; substituting target distributions, comparison data, sample counts, output coordinates or evaluation metrics; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. Sample counts and metrics are enforced by the validator and scorer; the rest rests on patch audit.

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
