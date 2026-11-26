# Randomized improvements of betting tests

## 1. Goal
Improve `power` for testing H0: E[X] = 1/2 from bounded iid observations.
The experiment uses Beta(20,b) samples, alpha=0.05, and randomized Markov
rejection of permutation-averaged betting capital. Candidates are compared with
the supplied LBOW/predmix baseline under the same observations and guards.
Candidate prefix-based working models, auxiliary fitting losses, training
procedures and predictable betting strategies may be replaced for this same
mean-testing problem. The observed data, capital arithmetic, randomization and
statistical guards are fixed.

## 2. Data
- `data/train/README.md`: no cross-instance training observations are supplied.
  Use validation only for evaluation and method, hyperparameter or checkpoint
  selection. Do not fit reusable models or learned preprocessing on validation.
  Within each isolated permutation, fitting to its own revealed prefix is allowed.
- `data/val/input.npz` and `data/val/truth.npz`: 9,500 public samples, with 50
  repetitions per alternative cell and 500 per null cell. Public seed: 20261030.
- The grid is b=19,19.2,...,20.8 and ten lengths from 100 to 2000,
  `numpy.linspace(100,2000,10,dtype=int)`. Inputs contain opaque IDs, lengths and
  zero-padded observations; labels contain the parameter and scoring cells.
- `bench/generator.py` freezes the original NumPy Beta generator.
- The normal predict command receives `data/val/request.json` or its private-test
  counterpart and exports the trained policy. The trusted scorer supplies the
  original observations sequentially during private execution. The public NPZ
  remains available for evaluation and method selection; no sample is replaced
  or resplit.
- Use only supplied data. Do not generate extra datasets, relabel inputs or use
  external data. Never memorize sample answers or reconstruct hidden seeds.

## 3. Metric and scoring
For each sample, average the terminal betting capital over the globally chosen
number of permutations, then reject iff `e_value > u / 0.05`. `power` is the
unweighted mean of the rejection rates over alternative parameter cells.
A valid method receives zero if null type-I error exceeds 0.0695 at any sample
length or 0.0562 pooled. Invalid artifacts, nonfinite capital or illegal access
sequences fail the submission. These absolute statistical guards are unchanged.

The scorer computes every capital and decision. It accepts arbitrary stateful
bet selectors through `code/policy.sh`; bets must be committed before the next
observation is revealed. Each permutation starts a fresh isolated process with
private temporary storage. No state or observations are shared with another
permutation or sample. A worker knows the current sequence length, null mean and
alpha, and can retain its entire observed prefix. It never receives future
observations, sample identity, permutation seed, the final U or truth labels.
The policy, permutation count, mixture weight and clipping scale are fixed before
private evaluation. A strategy may adapt its bets to the legal observed prefix.

The baseline validation power is 0.47281481, with seed standard deviation
0.00027716, measured using the certificate artifact path. Evaluation uses three
algorithm seeds; the relative improvement threshold is
max(0.02, 2*sigma/reference_mean).

## 4. Baseline
Official code: https://github.com/tmanole/Randomized-Markov at
`deff085f49f7ef3db0ce0899dac1ce4284d71f55`. The baseline uses 100 in-place
permutations, confseq 0.0.11 positive LBOW and default negative predmix strategies,
theta=0.5, trunc_scale=0.5 and maximum combination. These strategy choices are
optional. The negative strategy retains the library's actual implementation.

The supplied train command writes this policy in `model/betting.json`. Its
`certificate` execution mode uses the frozen NumPy arithmetic directly, preserving
the original baseline and seed schedule. `stateful` mode runs editable candidate
code instead and supports arbitrary histories, learned parameters and algorithms.
The existing expression grammar is an optional fast representation, not a limit
on candidate methods. `code/policy.py` demonstrates the stateful callback interface.

Across three reference repetitions, baseline training took 0.62 seconds each.
Validation artifact export took 0.82, 0.87 and 0.92 seconds on 16 CPUs. These
timings exclude trusted scoring and do not measure stateful execution. Process
startup and one interaction per observation may be expensive.

## 5. Interface
- `bash code/train.sh --out DIR` saves the global method configuration and any
  learned artifacts under DIR, before private evaluation observations are released.
- `bash code/predict.sh --model DIR --input PATH --output FILE` exports an inert
  `private-policy-v2` artifact. The supplied adapter uses
  `bench/private_eval.py::export_artifact`; it does not fit on evaluation inputs.
- A stateful artifact includes `code/policy.sh --model DIR --input REQUEST
  --output FILE`. The worker uses `private_rpc.call('bet', {...})` to commit bets
  and obtain one observation at a time. See `bench/io_spec.md` for exact fields.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and
  `BENCH_CODE_DIR`. CPU budget: 16; memory budget: 16 GB. No network is available.
- Use `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`
  and `bench submit --final` for development and submission.
- Dependencies use reproducible exact PyPI pins in `code/env.lock`. Install with
  `uv pip install`; editable, local and git dependencies are unsupported. The
  baseline requires NumPy 1.26.4.

## 6. Rules and pitfalls
Everything under `code/` may change while respecting the observation contract and
fixed statistical problem. `bench/`, supplied data, README, script signatures and
scoring rules are frozen. Do not read hidden labels, infer parameters by a truth
table, choose a global policy from evaluation samples, pass observations between
permutations, bypass the betting channel or forge completion before all bets.

The same initial observations are shuffled in place using the original reset
schedule; they are not independently regenerated for each permutation. A worker
must use the supplied horizon but only its own revealed prefix. All method state
must start from the saved global model in each new permutation. Do not cache state
in external files or processes. The trusted scorer owns the capital, U and final
rejection decision; the candidate never supplies those as self-reported results.

## Research contract

**Objective:** Improve power of the fixed bounded-mean test constructed from permutation-averaged predictable betting capital and randomized Markov rejection.

**Fixed research objects:** The null E[X]=1/2 for bounded iid observations, supplied sequences, alpha=.05, trusted capital recursion and final randomized Markov rule remain fixed. Bet-before-observation predictability, known horizon n, fresh state per permutation, same permutation/U schedule and all per-n/pooled type-I thresholds remain fixed.

**Editable methods:** Any stateful predictable bet selector, prefix-based working model, learned parameters and auxiliary loss may replace LBOW/predmix. Global permutation count and admissible mixture/clipping configuration may be chosen before private observations, within the fixed capital construction, as part of a changed bet selector.

**Forbidden substitutions:** Do not substitute a parametric Beta-only null or generic p-value classifier, self-report e-values/decisions, change the capital arithmetic or condition bets on future observations. No sample ID/truth/U/shuffle-seed access, cross-permutation information sharing, evaluation-driven global policy selection or altered sequences. When retaining the supplied betting_interface.DEFAULT pipeline, changing only B=100, positive="lbow", negative="predmix", theta=0.5, trunc_scale=0.5 or combine="max" does not count as a method improvement.

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
