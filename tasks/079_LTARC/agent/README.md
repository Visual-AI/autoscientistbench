# Learning Treatment Allocations with Risk Control Under Partial Identifiability

## Goal and fixed experiment

**Research question.** A policy assigns treatment 0 or 1 from covariates, and a binary
loss marks a non-beneficial outcome. Population risk is the share of losses under the
policy. Treatment risk is the share of losses among the treated, that is, patients
exposed to treatment without benefit. How low can population risk go, within an
interpretable policy class, while treatment risk stays at most a tolerance tau,
certified from finite samples with probability at least 90%?

**The paper's method.** One data part learns a tree for each nominal tolerance:
minimum empirical population risk, with empirical treatment risk below that tolerance.
A second part bounds each tree's treatment risk from above and selects the lowest
estimated population risk among trees whose bound is below tau. A third part evaluates.

**The experiment.** STAR is a randomized trial of class size. The paper varies tau over
100 random splits, without a competing method: low tau gives treat-none, high tau gives
treat-all, and at 0.35 it learns depth-two trees that treat part of the population.
This task fixes tau = 0.35 and compares a candidate with the baseline on the same
partitions under the same guards.

The private experiment retains the original 100 repetitions, all 4218 preprocessed
STAR students, 18 covariate columns encoding the paper's 11 covariates, the original
whole-cohort preprocessing, and seeds 10603+i for i=0,...,99. Each repetition has 2109
policy-learning, 1054 certification and 1055 evaluation rows. A student may appear in
different roles across repetitions; those repetitions execute separately. Treatment 1
denotes small class, treatment 0 regular class; loss 1 denotes a three-test total at or
below the original cohort median.

One tree per repetition may contain at most three single-variable conditions.
Continuous thresholds use the released training-derived 40-bin grid, including its
exact percentile positions and original prediction edge behavior. Gamma = 1 and
propensity = 0.5 are fixed. Gamma is the paper's degree of miscalibration; at 1 its
risk bounds hold with equality.

## Data and isolation

`data/train/` is empty: this task has no separate training split. A tree is fitted only
in a repetition's fit phase, on that repetition's learning rows.

Public `data/val/` is the development set: real patients from the International
Stroke Trial (IST), the paper's other real dataset. It shares no rows with STAR and is
entirely public, including its labels. It is for evaluation and for method,
hyperparameter or checkpoint selection; no model or learned preprocessing fitted on it
may enter your code. `cohort.npz` holds 18298 patients after the paper's exclusions:
9154 without aspirin and 9144 with aspirin. Treatment 1 is aspirin and 0 is no aspirin;
heparin is a covariate. Loss 1 is death at six-month follow-up. The paper's 23
covariates become 53 columns: 4 numeric columns on the same 40-bin grid, 7 binary
columns and one Boolean column per value of the 12 categorical covariates. There are 40
repetitions, seeds 10603+i for i=0,...,39, each with 9149 learning, 4574 certification
and 4575 evaluation rows. Gamma, propensity, the policy class, the worker roles and the
scoring rule are those of the formal run, the judge's replay on STAR.

Tau on IST is 0.225 in place of 0.35, in every guard. Treating every IST patient has a
treatment risk near 0.22, so 0.35 would never bind there. 0.225 is a tolerance from the
paper's IST experiment at which treating everyone can rarely be certified, as 0.35
cannot on STAR.

A score on IST is a guide to whether a method change helps, not an estimate of the
hidden STAR score: the patients, covariates, risk levels and tolerance all differ.

Each repetition first runs candidate `policy.sh` in a fresh sandbox with only that
repetition's learning rows. The worker commits a list of tree certificates. The
trusted evaluator verifies every certificate against the learning-derived grid.
A separate fresh sandbox receives certification rows and the immutable list, and
may return only an index into that list. It cannot add or change a tree. Evaluation
features and labels remain in the trusted scorer, which reconstructs all decisions.
No worker receives another repetition's rows, shared writable state, or prior logs.
The outer process receives aggregate scores only. Candidate error text is redacted.

## Scoring

For each repetition, population risk is mean `2*y*I(decision=a)` over evaluation rows.
Treatment risk uses the same weighted loss among treated decisions. The released edge
conventions are retained: treat-none gives treatment risk zero; positive treatment
with zero estimated treatment loss gives treatment risk one. Coverage is the fraction
of repetitions with treatment risk at most 0.35.

The score is mean population risk when coverage >= 0.90 and mean treatment risk
<= 0.35; statistical guard failure returns the specified score **1.0**. An invalid
archive, invalid tree, illegal certification response or failed replay returns
`Infinity`. One failed worker, or an exhausted evaluation time budget, fails the whole
replay. Structural validity and information boundaries are enforced by the trusted
scorer. The statistical validity of a replacement certification procedure rests on
patch audit; arbitrary selection is not a valid replacement.

## Baseline

`code/method/` holds the released FFT method modules, called from `code/adapter.py`. The
fit phase runs the original greedy search at lookahead 0 and 1 over 100 lambda values
from 0.01 to 0.8, where lambda is the nominal tolerance, and commits the best tree of
each run. Lookahead 0 yields only constant policies; lookahead 1 adds one split at a
time. The tree search adds 0.02 per split to its objective (`loss_i + 0.02 * n` in
`code/method/create_fft_constraint.py`). Certification uses a Hoeffding-Bentkus upper
confidence bound, with bound zero for treat-none.

## Research contract

**Objective.** Minimize certified STAR population risk at tau = 0.35 within the original
interpretable fast-and-frugal policy class.

**Fixed research objects.**

- STAR outcomes and treatment meaning, preprocessed covariates, the three data roles
  and the 100-repetition aggregation (scorer).
- The policy class: one tree per repetition, at most three single-variable conditions,
  the 40-bin grid (scorer).
- Gamma, propensity, the released inverse-propensity risk estimators and both guards,
  including the zero-loss edge cases (scorer for scored values; patch audit for a
  worker's own estimates).
- The 0.02 per split penalty in the tree search objective. Do not change, tune, schedule
  or remove it, and do not offset it by rescaling the loss it is added to. A gain must
  come from the search, the auxiliary loss or the certification, not from a cheaper
  split (patch audit).
- Any search or auxiliary loss that generates candidate trees keeps the same 0.02 per
  split in its objective; committing trees from an objective without it counts as
  removing the penalty (patch audit).
- The certification confidence stays at 90% (`self.delta = 0.1` in
  `code/method/weights_p_bound.py`); a replacement bound must hold at that level
  (patch audit).

**Editable methods.** Search over the same policy class, auxiliary learning losses,
optimizers, the lookahead and search strategy, and valid certification methods may
change, within the fixed policy class and information roles. The baseline's greedy tree
builder, lambda grid and Hoeffding-Bentkus bound do not fix the solver. Changing only
`lookahead`, `lambda_range` or `lambda_n` in `code/adapter.py` does not count as a method
change (patch audit).

**Forbidden substitutions.**

- Deeper trees, forests, neural or oblique policies, or another threshold grid (scorer).
- Another causal estimand, treatment meaning, propensity or outcome labels (scorer for
  scored values; patch audit otherwise).
- Fitting tree structure, thresholds or leaves on certification outcomes, which may only
  select among committed trees (scorer). Evaluation labels may never influence fitting
  or certification (scorer).
- External data, including STAR records, reconstructed labels, memorized policies or
  hard-coded outcomes (patch audit), or cross-repetition state (sandbox).
- Fitting reusable state or learned preprocessing on validation, including final
  train+val refits or validation-based distillation (patch audit).

## Entry points

- `bash code/train.sh --out DIR` prepares method settings; real fitting occurs in private replay.
- `bash code/predict.sh --model DIR --input PATH --output FILE` exports an inert policy archive.
- `bash code/policy.sh --model DIR --input DIR --output FILE` is invoked only by private replay.
- `bench/validate.py PRED INPUT` checks the inert archive; `bench/scorer.py PRED LABELS` executes replay.
- `bench eval-dev` runs these calls on the IST development set, through the same private replay the judge uses on STAR. For the baseline this takes about 2 minutes on one CPU.

`BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`
are provided by the harness. See `bench/io_spec.md` for phase schemas.
The current evaluator runs independent repetitions
sequentially, with one CPU and one numerical-library thread per worker. Read `names`,
`boolean` and `tau` from each phase's `request.json`, because they differ between the
development set and the formal run.

## Environment and editable surface

Python 3.11 and the five exact pins in `code/env.lock` are required. The trusted
scorer uses its reference environment, which baseline workers reuse read-only.
If candidate pins differ from the frozen reference pins, the archive exports the
locked candidate distributions for isolated workers. NumPy, pandas, SciPy,
python-dateutil and six suffice for the baseline.

Edit `code/` and dependency pins. Data, `bench/`, the certificate schema, metric,
fixed research objects and entry signatures are frozen. No model-class or isolation
bypass is authorized by an otherwise valid output archive.

## Reading results

The judge compares mean population risk on STAR with its archived baseline. Compare a
change on IST with the IST baseline from `bench status`, not with a risk on another
dataset.

On IST the baseline is close to a constant policy: in 29 of the 40 repetitions it
treats no one and in the other 11 it uses a one-split tree. Its population risk there
is 0.2354, against 0.2348 for treating no one. The fixed split penalty is one reason:
the two IST arms differ by about 0.011, less than one split's penalty. Its treatment
risk is 0.0503, its coverage 0.95, and its population risk has standard deviation
0.0083 across the 40 repetitions.

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
