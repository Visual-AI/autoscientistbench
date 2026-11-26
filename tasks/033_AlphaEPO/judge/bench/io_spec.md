# Curve prediction interface

Each input CSV is a set of queries for the fixed artificial-network setting:
with self-anticonformity, alpha=0.1, q=3, ordered initial condition, N=320,
empirical target mean degree 14.81 and clustering 0.49. The WS generator uses
ring degree 14 and beta=0.115. The baseline uses PA degree 14.81.

Input header: `id,p`. `id` is an opaque alignment key. `p` is the public
independence probability, between 0 and 0.50. Full curves use step 0.01.
Rows are shuffled; output may use any order. Neither ids nor row order have
physical meaning. Seeds, graph realization identities and MC observations
are absent. These are public experimental queries, not unknown parameters.

Preferred output header: `id,pred`. One row per input id, no duplicate or extra IDs. `pred` is finite c_S in [0,1] (roundoff tolerance 1e-8). Legacy `id,pred,cpu_s,analytic` is also accepted: `cpu_s` is finite nonnegative process-tree CPU seconds, repeated identically on all rows; `analytic` is a legacy method-description flag in {0,1}, also identical across the curve. Runtime is diagnostic. Explicit analytic=0 violates the analytic-approximation requirement and is rejected; analytic=1 or omission still requires source review. The raw MAE formula is unchanged.

Labels header: `id,p,y,mc_se,baseline_pred,baseline_mae_se`. `y` is the mean raw
c_S over independent MC runs; MC uncertainty uses graph-cluster bootstrap.
The remaining columns support evaluation diagnostics only. No fitting to MC
labels, baseline outputs, or simulated tables is allowed.

`python bench/validate.py PRED INPUT` exits nonzero for invalid format.
`python bench/scorer.py PRED LABELS` prints one JSON line with `mae`.
Valid outputs score mean absolute error over p. Malformed files score Infinity or fail validation. Output values alone do not prove lawful data provenance; see the README.

## Research scope

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
