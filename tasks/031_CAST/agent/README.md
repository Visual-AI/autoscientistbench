# CAST: Cross-Asset State-Space Trading

Improve the mean annualized Sharpe ratio across NASDAQ, CSI300, TPX100 and Global30.
Each panel has 30 independent accounts initially funded with 1000. Compare causal
trading methods on the same chronological observations, calibration inputs,
accounting and risk guards. Observations arrive sequentially through the scorer.

## Data and protocol

- data/train/input.csv contains calibration prices from 2005–2007 (754, 745, 765
  and 781 rows respectively). data/val/input.csv contains public 2008–2009 prices
  preceded by calibration history; validation block lengths are 505, 490, 488 and
  521 closes. data/val/labels.npz contains public scoring information.
- Global30 retains normalization by its first calibration close. Other panels
  retain original scales.
- Outer prediction gets only request.json. Full hidden histories and future
  prices are not passed to candidate code. The scorer starts one isolated policy
  worker per panel, releases one close, waits for committed orders, then releases
  the next close. Warm-up and final settlement rows carry active=false.
- No external data, relabeling, historical-path lookup or date-related hardcoding.
  Validation is for evaluation and method, hyperparameter or checkpoint selection.
  Do not fit reusable models or learned preprocessing on validation observations.

## Scoring and execution

train.sh fits calibration parameters only on data/train/input.csv and uses
validation only to select configuration. It saves all required model artifacts.
predict.sh exports the candidate code/model artifact. validate.py checks archive
structure without executing code. scorer.py uses the artifact's policy.sh in an
isolated process and records decisions through private_rpc.call.
See bench/io_spec.md for the event contract. A supplied order CSV is rejected.

The unchanged accounting.py reconstructs next-close fills, independent accounts,
no transfers/costs, decision-close no leverage, and permanent liquidation below
NAV 200. Positive NAV is mandatory. Daily mean account NAV gives simple returns;
Sharpe is sqrt(252)*mean/std(ddof=1), averaged equally across four panels. MDD
must be at most the corresponding frozen baseline plus 0.01, and turnover at most
twice its frozen baseline. Zero-variance returns and violated guards fail with
Infinity. Turnover includes zero-order days and excludes terminal settlement.

A forecast for t+1 may use prices through t only. The scorer withholds later
observations. Missing, nonfinite, duplicate or out-of-order responses fail.
Models, calibration estimators, controllers, losses and optimization procedures
may be replaced. Every method must use only legally available observations and
obey the same trading/risk rules.

Measured validation Sharpe is 0.7498466812757798, with standard deviation 0.0
from one reference run. All validation guards passed. The required relative
improvement remains 0.1%.

## Baseline and editable surface

Official source: https://github.com/FanBroWell/CAST, upstream commit
2d0ff46104611a47940be11db73717a663a0457a. The baseline combines Kalman forecasts
with weights based on recent forecast errors and couples assets through
calibrated price-change correlations. Its MPC optimizes predicted profit minus
an uncertainty-weighted trading penalty, commits the first action and replans
at the next close. Code is in code/src/filter, code/src/mpc.py and
code/src/backtest.py. code/policy.py processes each observed close in this order:
settle previous orders, liquidate margin stops, update filters, solve MPC and
clip orders.

Calibration uses the original per-asset 4x4 grid and rho from calibration price
differences. c is selected from {1,1.5,2} by mean validation Sharpe across panels
and lambda cells; each panel then selects lambda from {0.05,0.1,0.3,0.6}. Ties use
lower MDD then original grid order; nonfinite scores rank last. Filters use IRW
orders 1,2,3, horizon L=7, beta=0.5 and margin fraction 0.2. These are baseline settings, subject to the limit in the research scope below. Alternative controllers and online updates from already observed prices are allowed; future prices and hidden scoring information remain unavailable.

Everything under code/ is editable within the information-access, accounting and risk rules; the MPC itself is replaceable.
Data, bench/, runtime boundary, README, metrics and scientific instances are
frozen. Preserve train.sh, predict.sh and policy.sh arguments. The baseline uses
Python 3.11, numpy 2.4.6 and scipy 1.17.1; honor BENCH_SEED/BENCH_CPUS and retain
rebuildable dependency pins. No network is available to the policy worker.

Up to four isolated panels run concurrently within the allocated CPUs.
CPU allocation is 16 and the configured memory budget is 16 GB. Measured training
took 99.05 seconds. Exporting the validation artifact took 0.72 seconds; the
validation policy replay took 29.94 seconds. Artifact export is separate from
policy execution. Use bench status, bench eval-dev, bench validate-output and
bench submit.

Orders are share changes executed at the following close. Initial history only
warms filters. The final close settles orders without another decision. Workers
receive observations through RPC; do not assume a full evaluation CSV exists.



The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


## Research scope

**Objective.** Optimize causal trading decisions for the four existing market panels, maximizing mean annualized Sharpe under the original account and risk constraints.

**Fixed research objects.** chronological market observations and calibration/validation split; 30 independent accounts per panel with initial cash 1000; next-close fills, no transfers/costs, decision-close no leverage, permanent stop below NAV 200 and positive-NAV requirement; daily mean-account NAV, original Sharpe/MDD/turnover definitions and frozen panel risk limits.

**Editable methods.** causal price forecasting, state estimation, trading controllers and derived features from permitted observations; training surrogate losses, calibration estimators, online state updates and optimization within the same accounting rules.

**Forbidden substitutions.** substituting market paths, dates, assets, account structure or execution/risk rules; future-price access, date-related answer tables, transfers between accounts or alternate return/turnover definitions. When retaining the supplied CAST pipeline, changing only L=7, IRW orders (1,2,3), beta=0.5, the c candidates {1,1.5,2}, the lambda candidates {0.05,0.1,0.3,0.6} or the per-asset 4x4 calibration grid does not count as a method improvement.

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
