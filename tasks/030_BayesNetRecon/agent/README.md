# A Bayesian approach to out-of-sample network reconstruction

The paper asks whether past snapshots of an evolving network, turned into an
empirical prior, can predict the next snapshot out of sample. This task keeps one
branch of its real-data experiment: the Bayesian Erdos-Renyi model (BERM) with
observed updates on the weekly eMID interbank panel. Each week the model predicts
next week's expected link count from the last observed link count, a Beta prior
calibrated on past weeks and next week's known bank count.

The experiment compares a candidate's prior calibration with the baseline's rolling
three-year empirical prior, through the same posterior rule on the same weeks.
Improve `re_l` (lower is better). The paper's Bayesian fitness model, self-sustained
updates and link-level scores are not part of this task; the data holds weekly
counts and density estimates only.

## Data and scientific protocol

- `data/train/weekly.csv`: 156 calibration weeks, 1999-W01 through 2001-W52, with
  columns id, N, L and V. N is the bank count, L the link count and V=N(N-1)/2 the
  pair count.
- `data/train/pstar.csv`: columns label, kind and p_star. It holds the published
  point estimates (L/V) and 763 jackknife replicates for those weeks. Later weeks
  have no jackknife replicates.
- `data/val/input.csv` and `labels.csv`: 104 public validation forecasts in 2002
  and 2003, with calibration history. `forecast=1` marks a scored week. They support
  evaluation and method, hyperparameter or checkpoint selection only.
  `data/val/replay/` holds the same two files for the scorer.
- The hidden chronological block and its preceding history are held only by the
  scorer. Each forecast sees previous observations plus its own id, N and V. Its L
  becomes an observation only after its forecast has been submitted.
- Outer prediction receives `data/val/request.json` or the equivalent hidden
  descriptor, not the chronological evaluation table.

Do not re-split, relabel, retrieve outside series or memorize answers.
Fit reusable model state and learned preprocessing only on designated training
observations. Public validation features and labels are for evaluation and method,
hyperparameter or checkpoint selection only. Online calibration follows the causal
feedback rule below.

Observe events are this task's causal feedback. A policy may keep every observation
released earlier in the same replay and calibrate on it, as the baseline's rolling
prior does; no lookback truncation is imposed. The private worker has no dataset
path, network, parent process view or persistent state from another worker, so
anything it needs from `data/train/` must be saved under `--out`.

## Execution and scoring

`train.sh --out DIR` fits and saves the model. `predict.sh --model DIR --input PATH
--output FILE` exports the code/model policy artifact using `export_policy.py`. The
scorer starts an isolated `policy.sh` process and supplies observations and forecast
requests through `private_rpc.call`. A response must be committed before the next
event is released. See `bench/io_spec.md` for the complete event and artifact
contract.

For each forecast, submit the Beta prior parameters `alpha,beta` calibrated from
permitted history. They are the only submitted values that affect the forecast. With
L and M=N(N-1)/2 from the latest observed week and V from the requested week, the
scorer computes:

    p = (alpha+L)/(alpha+beta+M), clipped to [1e-9, 1-1e-9] as in the original solver
    predicted_L = V*p
    re_l = mean(abs(predicted_L-observed_L)/observed_L)

The mean gives every forecast week of the block equal weight, and every observed L
is positive. The measured baseline on the development block is `re_l` 0.079427393375
with std 0.0, because the method is deterministic. The relative improvement
threshold is 1%.

Guards and how they are enforced:

- Scorer: exact requested identities, complete chronology, and finite positive
  `alpha` and `beta`. An optional `p` must match the trusted posterior, and free
  `pred` values are rejected. A second `next` before the current reply is rejected.
  Invalid execution, missing replies or attempts to prefetch future information
  return `Infinity`.
- Validator: archive structure, safe paths, task identity and bounded size, without
  executing the artifact. A forecast table is not a valid submission.
- Patch audit: the scorer verifies posterior arithmetic, not the source of the
  prior. Prior provenance, algorithm-seed compliance and the prohibitions on
  memorized or external answers are checked by reading the code.

## Baseline and editable surface

Official source: https://github.com/mattiamarzi/OR4CLE, upstream commit
194e9bc52028cca96a4561f3b0152c23c0d71e8c (MIT).
`src/or4cle/BERM_solver.py` remains unchanged. `policy.py` is the streaming form of
the original batch loop in `main.py predict`, which is kept but not used for scoring.

For each forecast the baseline refits the prior by method of moments on the original
inclusive three-calendar-year window ending at the last observed week. It keeps the
point/jackknife sample order, and weeks after the calibration period contribute only
L/V. On all calibration weeks this fit gives alpha 58.261 and beta 541.543, the
values in the paper. The Beta branch is exact and deterministic; its retained GH
settings are 25 nodes, 60 iterations and tolerance 1e-10.

Prior-parameter calibration, lawful window selection and numerical fitting under
code/ may change within the fixed BERM/Beta model and causal event contract. Data,
bench/, README, runtime boundary, metric and scientific instances are frozen.
Preserve `train.sh`, `predict.sh` and `policy.sh` arguments and write model artifacts
under `--out`. The output of predict must be an artifact, never a precomputed table
of evaluation forecasts. Never rely on raw hidden CSV paths or current/future L being
present.

Python 3.11 and numpy 2.4.6 are the baseline dependencies. Honor BENCH_SEED and
BENCH_CPUS. The worker uses at most 8 allocated CPUs, with thread counts bounded by
its actual allocation; the configured memory budget is 4 GB. At 8 CPUs, the measured baseline
takes 0.51 seconds to train and 0.51 seconds to export the policy artifact.
Validation input loading and policy prediction take 0.52 seconds, excluding
artifact preparation. Frozen helper scripts are imported through BENCH_BENCH_DIR, and code uses
explicit PYTHONPATH because the harness sets PYTHONSAFEPATH=1.

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


## Research scope

**Objective.** Improve one-step observed-update Bayesian Erdos-Renyi network
reconstruction on the supplied weekly panel using the original Beta-prior
posterior-predictive mean and mean relative link-count error.

**Fixed research objects.** These define the paper's experiment, so that a score
change reflects the prior calibration alone.

- Existing chronological weekly observations, node/pair counts and
  development/evaluation partition (read-only data; the judge uses its own copies).
- BERM shared edge probability, Beta prior family, observed-update
  posterior-predictive mean and next-week expected count `V*p` (the scorer computes
  these; the prior family behind `alpha,beta` is patch audit).
- Forecast-before-observe event order and original all-week mean relative error
  (scorer).

**Editable methods.**

- Beta hyperparameter calibration from permitted past observations, lawful
  calibration-window selection and numerical estimation procedures.
- Implementation, internal representations and optimization of prior fitting within
  the same Bayesian statistical model.

**Forbidden substitutions.** The prior must be fitted to the distribution of past weekly density estimates L/V, as the paper's empirical prior is. Do not choose alpha and beta by minimizing forecast error, scale their sum by a free concentration factor, or use fixed or near-zero values that remove the prior or the observed week from the posterior.

- Replacing BERM/Beta posterior reconstruction with an unrelated discriminative
  forecaster, a different prior family, posterior mode or different network model
  (a directly submitted forecast or posterior mode is rejected by the scorer; the
  rest is patch audit).
- Reverse-solving Beta parameters from desired predictions rather than calibrating
  a prior on allowed history (patch audit).
- Future information (scorer) or memorized weekly answers (patch audit).
- Training or fitting reusable model state or learned preprocessing on validation
  features or labels, including final train+val refits or validation-based
  distillation (patch audit).

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
