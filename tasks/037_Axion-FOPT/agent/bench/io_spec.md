# Axion inputs and expression certificate

Input CSV has exactly `id,theta0,hstar,beta_over_h,vw`. Training and label CSVs
append exactly `xi,xi_sem`. Angles are radians in [0,pi), hstar is H*/M_phi,
beta_over_h is beta/H*, and 0 < vw <= 1. All numbers are finite, rates positive,
xi positive and xi_sem nonnegative. IDs are opaque strings.

Prediction is UTF-8 JSON at the requested filename, at most 16 MiB, with exact keys:
`format` = `axion-expression-v1`, `model`, and `rows`. Each row has exactly
`{"id": input_id, "xi": positive_number}` in input order. The model is shared
by every row and has exact keys:

- `format`: `axion-expression-v1`.
- `parameters`: an ordered list of zero to seven finite numbers, addressed
  as `p0`, ..., `p6`. These are the entire fitted numeric parameter channel.
- `pre`: an ordered list of `[name, expression_string]` assignments.
  It must define `kappa`, a finite nonnegative radius threshold.
- `post`: ordered assignments after physical geometry has been computed.
  It must define positive `xi_dm` and `xi`.

Each stage has at most 32 assignments. Each expression has at most 2,048
characters, 256 AST nodes and nesting depth 40. Names cannot overwrite inputs,
parameters, other assignments, functions or fixed primitives. The interpreter
supports +, -, *, /, **, unary +/-; unary abs, log, log1p, exp, expm1, sqrt,
tanh, sin, cos; and min/max with 2–8 arguments. Numeric literals are restricted
to structural 0,1,2,3. Fitted values must use the global parameter list.

The pre-stage scalar inputs are `theta0,hstar,beta_over_h,vw` and fixed physical
primitives `tp,tau,fanh,finf,pilot,gamma_vw,pi,e,tiny,sum_floor,power_floor,a_exp,xi_floor,p_mink,
kappa_eps,fanh_eps,fanh_alpha,finf_eps,finf_alpha`. Values come from the frozen
independent calibration and original source routines. Expressions receive no
IDs, row index, labels, file access, arrays or Python calls.

After kappa, frozen original quadrature supplies `f_bm,g_bm,rc_mean,a_ratio,
v_factor`. Here a_ratio = 1 + sqrt(3)/2*hstar*rc_mean/vw, and
v_factor = gamma_vw**p_mink/a_ratio**2. The distribution is computed with the
source adaptive radius support and original discrete threshold; equivalent
cumulative sums cache the independent distribution. Original fallback bank
interpolation is retained.

The baseline graph in `code/baseline.py` is editable. Candidate formulas and
fitting algorithms may differ, while their final expression certificate uses
these physical primitives and at most seven fitted scalars. Additional fitted
coefficients hidden in algebra, graph structure or fixed calibration are
forbidden; explicit parameter counting alone does not certify their provenance.
Source/provenance audit complements the numerical checks.

The scorer reconstructs predictions (rtol 1e-10, atol 1e-12), then performs the
fixed physical tests. At (hstar,vw,theta0,beta_over_h)=(.2,.5,1.2,6), xi, xi_dm,
rc_mean and tp must be positive, 0 < f_bm < 1, and g_bm >= 0. At
(hstar,vw,theta0,beta_over_h)=(5,.5,1.2,6), require 1 <= xi < 2. At
(vw,theta0,beta_over_h)=(.7,1,4), xi must strictly decrease across
H=1e-3,1e-2,.1,1, with xi(1e-3)>10. At (vw,theta0,beta_over_h)=(.7,.1,4),
the log slope between H=1e-6 and 1e-5 must differ from -.5 by less than .05.
The score is equally weighted natural-log RMSE on recomputed predictions.
Failures report `log_rmse=Infinity, raw_score=1.0`.

Only `data/train/train.csv` may enter numerical constant fitting. Validation is
only for evaluation and method, hyperparameter or checkpoint selection. Do not
fit models or learned preprocessing on its features or labels. The train entrypoint produces
`model.json`, exactly the model object above. A `fit_report.json` is diagnostic
only and is never trusted as a certificate. The expression certificate contains
all state that can influence trusted predictions.

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
