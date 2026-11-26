# Momentum SVGD-EM for Accelerated Maximum Marginal Likelihood Estimation

The judge rebuilds your code, runs it and scores the result on hidden images against the baseline. All rules below are binding.

## 1. Goal

**Research question.** Does acceleration improve predictions in a high-dimensional, non-convex model? The source paper (Rozzio, Athanasiades and Akyildiz, AISTATS 2026) adds momentum terms to SVGD-EM, a particle-based EM method for maximum marginal likelihood estimation, and calls the result M-SVGD-EM.

**Model.** A Bayesian neural network that classifies MNIST digits 4 and 9. The latent variables are `w` (40 x 784) and `v` (2 x 40). The class scores of an image z are `v . tanh(w . z)`, and the likelihood of a label is their log-softmax at that label. Every weight has a zero-mean Gaussian prior, with scale `exp(2*alpha)` for `w` and `exp(2*beta)` for `v`, and `log_prob` is (log-likelihood + log-prior) divided by the number of images. theta = (alpha, beta) is estimated by maximum marginal likelihood. The tanh likelihood is non-convex and has no closed-form posterior or marginal likelihood. A method maintains theta and a cloud of 10 particles, each one a (w, v) pair.

**Experiment.** Each trial runs 500 full-batch iterations on the same 800 training images, from a supplied initial state with theta = (0, 0) and standard normal `w` and `v`. The published comparison uses 10 independent trials and plots ensemble test error and log predictive density against the iteration: M-SVGD-EM with alpha 0.3, 0.5 and 0.9 ends below SVGD-EM, and alpha 0.9 is lowest. All methods there use the same optimiser and step size.

**Your task.** Given the training images and the supplied initial states, produce the final cloud and theta of each trial. Improve `test_error` (lower is better) against the baseline M-SVGD-EM(alpha=0.9) while satisfying the LPPD and oracle-budget guards. The hidden test is unseen MNIST images of the same two digits. Your code never receives those images: the scorer applies the frozen forward function to the `w` and `v` you submit. You may improve the method within the rules in §6 and the Research contract.

## 2. Data

- `data/train/train.npz`: `X` float32[800,28,28] and `y` int32[800], with 0 for digit 4 and 1 for digit 9. These are the only observations a method may fit on.
- `data/val/input.npz`: 10 validation trials with opaque `id`, initial states `w0`, `v0` and `theta0`. Their random draws use JAX keys 102..111, following the rule in `bench/io_spec.md`.
- `data/val/truth.npz`: the 200 validation images `X` and labels `y`, the keys, and the frozen baseline guard values. They are public and are for evaluation and selection only.
- The 800 and 200 images are the authors' fixed split (`random_state=0`). Images are already standardised per pixel with a scaler the benchmark computed once on these 1000 images, following the authors' preprocessing, and pixels that are constant on them are 0. The scaler is part of the fixed data, so no method fits it and you may not change it.
- The hidden test images are held by the judge and cannot be read from this workspace.
- Do not re-split or relabel the data, and do not add images from any source, including other MNIST images and copies from an installed package. Each trial must depend only on its own input row and the training images. Never memorize IDs.

## 3. Metric and scoring

The score is the mean over 10 trials of `test_error`: the fraction of images misclassified by the trial's final cloud, where the predicted class is the argmax of the log-softmax averaged over the 10 particles. The lower bound is 0. The scorer computes it from your `w` and `v` with `bench/bnn_metrics.py`, which is the authors' function, unchanged.

| Guard | Enforcement class | Rule |
| --- | --- | --- |
| Predictive quality | scorer | The mean over trials of the authors' `log_pointwise_predrictive_density` must be at least the same-images frozen baseline mean minus 2 population SDs over baseline trials. |
| Declared per-iteration oracle budget | scorer | The unit is one particle evaluated on one image. In each of the 500 iterations, latent-score calls may cost at most 8000 and theta-gradient calls at most 8000; kernel derivatives are excluded. The baseline uses 8000 + 8000. |
| Actual oracle budget, full batch, truthful cloud | judge probe | An instrumented replay recounts model calls, requires every call to use all 800 training images, and compares the final state with your output. |
| Fixed predictor | scorer | Error and LPPD are computed from the submitted `w` and `v` with the frozen forward function. |
| Trial independence | judge probe / patch audit | Predictions must agree when rows are evaluated alone or in a reordered subset. |
| Initial state | patch audit | Start from the supplied `w0`, `v0` and `theta0`. The output does not contain them, so the scorer cannot check this. |
| Base update and particle momentum | patch audit | AdaGrad with step size 0.1 and its `ox.adagrad` defaults for both updates, and alpha=0.9 if the memoryless particle form is kept. See §6. |
| Reported cloud | patch audit | The state after the 500th update, with no post-processing. See §6. |
| Model, data, no hidden data, no fabricated states, no instrumentation bypass | patch audit | See §6 and the Research contract. The replay entry point must implement the same method as predict. |

`bench/scorer.py` implements the numeric rule literally: the feasible mean, otherwise **1.0**. Invalid output is rejected with **Infinity**. A failed replay rejects the submission.

Validation baseline: test_error **0.0740**, trial population SD **0.0159**, Monte Carlo SE **0.0053**; LPPD **-0.6625**, trial population SD **0.0438**, frozen limit **-0.7502**. The run is deterministic given its inputs, so the baseline's seed SD is 0. PASS requires a relative improvement on the hidden images of at least max(delta, 2 x baseline seed SD / baseline mean), and delta is **11%**.

## 4. Baseline

Official code: https://github.com/Farattel/m-svgd-em, commit `f6c7159623d47a7868c01f0663d8007f9377d90b`. The baseline is M-SVGD-EM(alpha=0.9): AutoRBF h=5.0, AdaGrad with step size 0.1 for both updates, N=10, T=500, 40 hidden units. Each iteration makes one latent-score call and one theta-gradient call, each on all 10 particles and all 800 images. The model is in `bnn_model.py` and the metrics in `bnn_metrics.py`, both copied unchanged from the authors' notebook, so the prior scale is `exp(2*theta)` and `log_prob` is divided by n, as coded there. The unchanged optimization lives in `msvgdem/zoo.py`, `expectation_step.py`, `maximisation_step.py` and `expectation_maximisation.py`. The baseline theta update has memory: it computes `theta_new + alpha*(theta_new - theta_prev)`, where `theta_prev` is the theta input of the previous iteration. Its particle update has none, see §6.

The `bench-adapt` commit adds NPZ data entry, configuration save/load, explicit train/predict commands, seeds and CPU controls, and an import alias `coinem`. It removes the original experiment notebooks and figures. `algorithm.run_trial` calls the original optimizer with `metrics={}`, which leaves the update loop unchanged and stores no history. The reported cloud is the state after the 500th update. The authors' figure plots the state before each update, so its last point is one update earlier: on validation the error is the same in all 10 trials and the mean LPPD is -0.6630 instead of -0.6625.

**The kernel is inert in the baseline.** Particles flatten to 31440 dimensions. In the two trials checked, the smallest squared distance between two particles stayed above 57000 for all 500 iterations, so with h=5 every off-diagonal kernel value and kernel gradient is exactly 0 in float32. The baseline therefore behaves as 10 separate momentum-AdaGrad networks coupled only through theta. Particle interaction has no effect on its result.

The code base also contains a closed-form M-step: `optimal_theta` in `bnn_model.py`, `MarginalStep` in `maximisation_step.py` and the `marginal_*` methods in `marginal_zoo.py`. The baseline does not use them. `optimal_theta` returns twice the maximizer of the coded prior: the theta gradient vanishes at `log(mean square)/4`, while it returns `log(mean square)/2`.

Runtime on 8 CPUs is **0.52 seconds** for train and **22.8 seconds** for validation prediction: about 8 to 16 seconds per trial while 8 trials share the CPUs, including compilation. `train.sh` writes `config.json`; `predict.sh` runs independent trials in a process pool of single-threaded workers. Each worker peaks at about 300 MB. The memory allowance is 16 GB.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: write all artifacts required by prediction.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write NPZ predictions with exactly `id`, `w`, `v`, `theta`, `oracle_counts`, as defined in `bench/io_spec.md`. It must work on any archive of the input format, with any number of rows.
- Environment: `BENCH_SEED` controls any algorithmic randomness; `BENCH_CPUS` is the total CPU budget; `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR` locate the available files. `predict.sh` reads the training images from `$BENCH_DATA_DIR/train/train.npz`.
- Keep `algorithm.run_trial(model, data, latent_init, theta_init, config, key, steps)` as the actual method implementation. It returns the final particle cloud (a dict with `w` and `v`) and theta (a dict with `alpha` and `beta`); the judge may replay this entry point with its trusted model and instrumented calls. The replay automatically tracks an outer `jax.lax.scan`; other loop implementations call `model.bench_set_iteration(i)` when available before each iteration's model queries, with zero-based sequential i. This hook only marks query accounting and supplies no extra observations.
- The replay counts model queries made through `model.score_latent_particles` and `model.average_score_theta`. It rejects any other direct evaluation of `model.log_prob` and any call whose data is not all 800 training images.
- Dependencies are pinned in `code/env.lock`. Editable, local and git installs are unsupported. JAX runs on CPU; no TensorFlow installation is needed for the pinned TFP JAX substrate.

## 6. Rules

The free surface is `code/`: change how the particles interact, how acceleration is done, and how theta is updated, within the Research contract. A changed implementation must report its actual oracle use and states. The data, `bench/`, this README, the I/O contract and the command signatures are frozen.

The base update of both theta and the particles is AdaGrad with step size 0.1, as in the published comparison, which holds it fixed for all methods. Do not change, tune, schedule or adapt the step size, do not replace AdaGrad, and do not emulate another step size by rescaling gradients or updates. Both AdaGrad updates keep the `ox.adagrad` defaults `initial_accumulator_value=0.1` and `eps=1e-7`; do not change, reset or rescale the accumulator. A closed-form M-step has no step size and is allowed; a gradient-based theta update uses AdaGrad with step size 0.1. In the baseline code the particle momentum term computes `new + alpha*(new - current)`, which is exactly a step of (1+alpha) times the AdaGrad update and has no memory; if you keep that form, keep alpha=0.9. Acceleration must come from memory of earlier iterates, for example the two-sequence form `x_tilde[t+1] = x[t+1] + alpha*(x[t+1] - x[t])` with alpha below 1, or from the theta update, the kernel or particle interaction.

Do not rescale `w` or `v` after the last iteration or otherwise post-process the cloud: report the state after the 500th update. Reporting an average of earlier iterates counts as post-processing.

Why these are fixed: the question is whether a method predicts better within a fixed number of full-batch iterations, so methods can be compared only on the same model, images, initial states, cloud size and iteration count, with the same cap on model evaluations per iteration and the same predictor.

The judge runs one seed because the baseline is deterministic given its inputs; honor `BENCH_SEED` if your method uses randomness. The judge uses 8 CPUs and 16 GB memory.

Disqualifying actions include reading or guessing anything the judge holds, embedding weights or results obtained before the run, using other images, fabricating states or oracle counters, and detecting or bypassing a replay.

## 7. Known pitfalls

Write NPZ through an open file handle: the requested filename may have no `.npz` suffix. Preserve ID order; missing, duplicated, reordered or nonfinite outputs fail validation.

Validation has 200 images, so one image moves a trial's error by 0.005 and the baseline's trial SD is about 0.016. Over 10 trials, a difference below about 0.011 is within trial noise. The hidden score has its own noise, so do not treat a small validation gain as a pass.

LPPD here is the mean log score over particles, as the authors coded it. It moves with the scale of the logits, while the test error does not. Check the guard on validation.

The oracle counter in the output is a declaration, not proof of cost. Keep it accurate and maintain a replayable method. A full unjitted replay of one 500-iteration trial takes about two minutes. Patch audit also checks for bypasses of the Python instrumentation.

Two different things are called alpha: `theta['alpha']` is a prior parameter, and `config['alpha']`, the 0.9 in M-SVGD-EM(alpha=0.9), is the momentum coefficient.

Results depend strongly on the starting point, so each comparison must use the same 10 initial states. Keeping the full history inside a JIT costs about 629 MB per trial; do not store it unless you need it.

The scripts export their package path for `PYTHONSAFEPATH=1`. Keep CPU workers single-threaded and respect `BENCH_CPUS`; unnecessary nested thread pools can exhaust the budget. Absolute command arguments must be resolved before changing directories.

## Research contract

**Objective:** Improve the ensemble test error of a particle-based EM method for the fixed Bayesian neural network on MNIST 4 vs 9, after a fixed number of full-batch iterations from supplied initial states, while keeping predictive quality and the oracle budget.

**Fixed research objects:** The tanh network with 40 hidden units, its likelihood, the Gaussian priors with scale `exp(2*theta)` and the division of `log_prob` by n remain fixed, as does the frozen predictor, which is the argmax of the log-softmax averaged over particles. Theta remains a quantity estimated from the particle cloud. The 800 training images, supplied `w0`, `v0` and `theta0`, 10 particles, 500 iterations, independent trials, at most one full-batch latent-score evaluation and one full-batch theta-gradient evaluation per particle per iteration, the test error definition and the LPPD guard remain fixed.

**Editable methods:** Particle interaction, kernels and their bandwidth, the form of the momentum or acceleration on particles and on theta, the M-step rule (gradient-based or closed-form) and implementation may change.
Not editable: AdaGrad with step size 0.1 and its `ox.adagrad` defaults for both updates (a closed-form M-step has no step size) and, if the memoryless particle momentum form is kept, alpha=0.9. See section 6.

**Forbidden substitutions:** Do not change the likelihood, the priors or the network, and do not swap the predictor or report anything other than this network's weights. Do not hold theta fixed, schedule it by hand or choose it by validation score. Do not use minibatches, split a full-batch evaluation into several updates or make uncounted model evaluations, including your own forward pass on training images for a line search or a stopping rule. Do not add data, augment images, add features or read validation images or labels while `train.sh` or `predict.sh` runs; settings chosen offline on validation are allowed. Do not discard the supplied initial state, warm-start from weights obtained before the run, or evaluate the model during `train.sh`. Do not rescale `w` or `v` after the last iteration or otherwise post-process the cloud: report the state after the 500th update. Do not fabricate states or bypass model-call accounting. No cross-trial fitting. No training or fitting of reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
