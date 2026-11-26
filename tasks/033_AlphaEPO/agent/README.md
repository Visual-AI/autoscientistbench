# Competition between private and expressed opinions: the alpha-EPO q-voter model

## 1. Goal

Predict the stationary positive expressed-opinion concentration c_S(p) for the
fixed alpha-EPO experiment with self-anticonformity: alpha=0.1, q=3, ordered
initial condition, and Watts-Strogatz networks matched to empirical S2
(N=320, target mean degree 14.81, clustering 0.49). Here c_S is the fraction of
sites expressing a positive opinion. Improve `mae` (lower is better) against
network Monte Carlo means. Methods may change within the research scope below.

Each site has a private and an expressed binary opinion. Both start positive.
An elementary update selects a site uniformly and updates its private opinion
with probability alpha, otherwise its expressed opinion. Independence is chosen
with probability p; otherwise a panel of q distinct neighbors supplies influence
through their expressed opinions. Under private independence, the opinion flips
or stays unchanged with equal probability. Private conformity follows a unanimous
panel, even if the site's expressed opinion differs. Expressed independence copies
the private opinion. Expressed conformity follows a unanimous panel when the two
opinions agree. When they differ, one panel member agreeing with the private
opinion is enough to make the site express it. These rules define the dynamics
that an approximation must represent.

## 2. Data

- `data/train/README.md`: no cross-instance training data. The baseline is training-free.
- `data/val/input.csv` and `data/val/labels.csv`: a public validation curve with
  51 queries p=0.00,0.01,...,0.50. Labels average 50 WS realizations × 2 runs.
  Public graph and MC seeds are in `data/val/seeds.json`.
- `bench/generator.py` and `bench/mc_generator.cpp` document the frozen data
  generation process. WS ring degree is 14 (the nearest even choice to 14.81),
  rewiring beta=0.115, calibrated from topology only. MC retains the authors'
  MCS=5000, thermalization=3000 and three-decimal raw output.
- Input contains only opaque `id` and the public query coordinate `p`. No
  graph identities, simulation seeds or opinion observations enter prediction.
  Labels include `y`, uncertainty and baseline diagnostics; these are for
  evaluation only. They must not fit or supply a simulated response table.
- Do not re-split, relabel, generate additional simulations or add external data.

## 3. Metric and scoring

`mae = mean_p(abs(predicted_c_S(p) - MC_mean_c_S(p)))`. The best possible value is
0. This MAE is the benchmark's comparison of the paper's observable, not a metric
printed in the paper. Raw c_S is used without magnetization folding.

| Guard | Enforcement class | Meaning |
|---|---|---|
| Valid concentrations, exact schema, finite values, exact id set | scorer / validator | Invalid files fail validation; direct scoring of malformed files gives Infinity. |
| Independent of ids and row order, reproducible under the same algorithm seed | patch audit | Renamed/subset queries must give matching predictions. |
| Analytic/ODE approximation only; no added simulation data, fitted evaluation-label tables or memorized outputs | scorer / patch audit | An explicit analytic=0 declaration is rejected. Other submissions still require source review of training and prediction data flow. |

The baseline on validation is **0.04973832686080737**, with algorithm-seed standard
deviation 0. The validation reference's graph-cluster bootstrap SE is
0.0010901964346982263. Bootstrap resampling retains both runs and the full curve
of each sampled graph. Differences smaller than twice the reference MC SE are
ties. The judge uses its own reference uncertainty; the relative PASS threshold
is about **3.5%** (exact value in `bench status`). The judge uses one seed.

Input-independence and lawful data provenance are checked by patch audit. Numeric predictions do not establish provenance.

## 4. Baseline

Official code: https://github.com/BarbaraKaminska/EPO-q-voter at
`fcc25134294d0eeabc311503b327773d0c857d0c`. The baseline uses pair approximation
at k=14.81, tracking correlations between neighboring sites. The released MATLAB
equations are in `code/PA_authors.m`; their translated arithmetic is in
`code/pa_rhs.cpp` and `code/pa_rhs_verbatim.py`. `code/pa_predict.py` integrates
the ten edge-state variables independently for each p with RK45, ordered
initialization, t=2000, rtol=1e-7 and atol=1e-9. These are baseline settings.

`code/train.sh` compiles a shared library and writes configuration; it fits no
labels. Prediction solves the approximation from the ordered state independently
at each query coordinate.

Measured with a 16 CPU budget: train **0.56 s**, validation prediction **8.58 s**.
The baseline uses one thread; unused CPU capacity is available to alternative
methods. The memory allocation is 4 GB.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes every artifact required by prediction.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with
  preferred header `id,pred`; legacy diagnostic columns are accepted. See `bench/io_spec.md` for all fields.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`,
  `BENCH_CODE_DIR`. Honor the seed and the CPU budget; the baseline equations
  themselves do not depend on the seed.
- Development commands: `bench status`, `bench eval-dev [--seed S]`,
  `bench validate-output FILE`, `bench submit [--final] [-m MSG]`.
- Install dependencies with `uv pip install`; submission freezes the environment.
  Rebuildable dependencies use `name==version` pins from PyPI. Python is 3.11;
  numpy and scipy are pinned in `code/env.lock`. A C++17 compiler is available.

## 6. Research scope and rules

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Reduce stationary expressed-opinion concentration MAE for the fixed alpha-EPO q-voter experiment using a deterministic analytic/ODE approximation.

**Fixed research objects:**

- Q-voter dynamics with self-anticonformity, alpha=.1, q=3, ordered initialization and the supplied S2-matched Watts-Strogatz setting.
- Original raw c_S(p), query grid, MC reference-generation protocol and uncertainty-aware MAE.
- Analytic/ODE approximation without prediction-time MC, additional simulations, or fitting/interpolating evaluation MC labels.

**Editable methods:**

- Analytical approximation, closure, numerical equations approximating the fixed physical dynamics, integration, solver and implementation.
- Configuration selection from public diagnostics for a changed method; PA at k=14.81, RK45 and its integration endpoint are baseline choices.

**Forbidden substitutions:**

- Replace the physical q-voter dynamics/network ensemble or raw c_S with folded magnetization or another observable.
- Replace analytic prediction with Monte Carlo, a simulation-trained regressor or evaluation-label lookup/interpolation.
- Change the MAE/uncertainty rule, infer hidden seeds or use IDs as physical information.
- When retaining the supplied pair-approximation equations, changing only k=14.81, the integration endpoint t=2000, method=RK45, rtol=1e-7 or atol=1e-9 does not count as a method improvement.


The analytic approximation and numerical implementation under `code/` may change within the research scope above. Preserve the experimental setting, target observable, data protocol and prediction interface. `bench/`, `data/` and this README are frozen.

The allocation is 16 CPUs and 4 GB memory. Prediction has no network access. Optional runtime diagnostics must be honest.

There is no training split. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Do not fit models or learned preprocessing on validation features or labels. Do not execute the shipped simulation generator to add data, fit or interpolate evaluation MC labels, embed simulated lookup tables, memorize paper curves or hard-code outputs. Do not read or infer hidden observations or exploit opaque IDs. Numerical work on the public equations and supplied query coordinates is allowed. Candidate randomness must honor `BENCH_SEED`, and predictions must be invariant to IDs and row order for the same seed and physical input.

## 7. Known pitfalls

- The strict author condition `mcs > MCS_term` averages 1,999 samples, not 2,000.
  The generated labels preserve this behavior and three-decimal output rounding.
- Finite-time ODE integration can relax slowly near the transition. The baseline
  uses the original t=2000 endpoint, with no convergence-based early exit.
- Empirical degree 14.81 and generated WS degree 14 are distinct. The baseline
  deliberately uses 14.81; do not confuse the input experiment with the PA closure.
- Write exactly the supplied output path, even if it has no extension. Convert
  NumPy scalars to Python floats before using `repr` in CSV serialization.
- All script paths may be absolute; the scripts export their package path because
  the execution environment sets `PYTHONSAFEPATH=1`.

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
