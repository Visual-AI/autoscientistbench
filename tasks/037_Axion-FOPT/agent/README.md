# Axion enhancement across first-order phase transitions

Predict the late-time axion abundance enhancement xi on unseen physical slices.
The supplied lattice targets and physical constraints are fixed.

## 1. Goal

Minimize natural-log RMSE for xi = rho_PT / rho_noPT from the initial angle theta0,
expansion rate H*/M_phi, transition rate beta/H*, and bubble wall velocity v_w.
Use at most seven globally fitted enhancement constants. You may change the
fitting method and the analytic enhancement predictor while preserving these
physical inputs, targets, independent auxiliary calibration and physical tests.

## 2. Data

- `data/train/train.csv`: 1,122 original lattice rows in complete physical slices,
  with inputs, xi and xi_sem. These are the only rows permitted for fitting
  enhancement constants in each training run.
- `data/val/input.csv` and `data/val/labels.csv`: 330 original rows in separate
  complete slices. Use validation only for evaluation and method, hyperparameter
  or checkpoint selection. Do not fit models or learned preprocessing on its
  features or labels.
- Preserve the materialized grouping by (H*,v_w).
- `bench/physics_data/`: fixed independent pilot-kappa, bubble geometry, Fanh,
  Finf and Minkowski velocity calibration. These are physical side inputs, not xi
  targets or additional enhancement fitting channels.
- IDs are opaque. The only target is xi; xi_sem is a diagnostic and is not a score
  weight. No external data, released fits or prediction tables may be added.

## 3. Metric and scoring

`log_rmse = sqrt(mean(log(xi_model / xi_lattice)**2))`, with natural logarithms and
equal row weights. Lower is better. The scorer independently evaluates an inert
model certificate and checks every submitted prediction.

| Requirement | Enforcement |
| --- | --- |
| Matching unique IDs and finite positive predictions | validator and scorer |
| At most seven global scalar parameters; permitted scalar expressions only | scorer |
| Fixed independent physical calibration and original radius quadrature | scorer |
| Prediction agrees with the certified model (rtol 1e-10, atol 1e-12) | independent scorer recomputation |
| At (hstar,vw,theta0,beta_over_h)=(.2,.5,1.2,6): xi, xi_dm, rc_mean and tp are positive; 0 < f_bm < 1; g_bm >= 0 | scorer |
| At (hstar,vw,theta0,beta_over_h)=(5,.5,1.2,6): 1 <= xi < 2 | scorer |
| At (vw,theta0,beta_over_h)=(.7,1,4): xi strictly decreases across H=1e-3,1e-2,.1,1 and xi(1e-3)>10 | scorer |
| At (vw,theta0,beta_over_h)=(.7,.1,4): log slope between H=1e-6 and 1e-5 differs from -.5 by less than .05 | scorer |
| Training-only fitting; no encoded targets or hidden fitted structure/constants | source and provenance audit |

A guard failure has mathematical rule score 1.0, reported as `raw_score`;
`log_rmse=Infinity` signals a rejected submission. Machine checks establish the
explicit scalar count and certified computation. They cannot prove that an
expression's structure, algebraically encoded literals, or initialization was
chosen independently of prohibited labels. Such hidden fitted capacity is
disallowed and requires source/provenance review.

The configured relative improvement delta is 0.025. Measured validation log-RMSE
is 0.1163004565875834, with standard deviation 0.0 from one reference run.
The baseline is deterministic.

## 4. Baseline

The original `xi_model` is from `baltabaygal/axion-fopt`, upstream commit
`1b67fecf3d40b707a9dd6f4c2eabefc8f9bb3f1b`. Its released package contained a global
fit but no routine to fit the seven constants. The benchmark therefore supplies
a documented reconstruction: bounded derivative-free Powell optimization of
training log-MSE, starting from neutral priors in `code/baseline.py`.
This is a new fitting protocol, not a recovered author optimizer.

The baseline combines delayed and bubble misalignment. Its parameters are ordered
C0,r0,r1,lambda,A0,s,q_k. The fixed fanh and finf are the no-transition and
late-percolation anharmonic corrections; tau is the rescaled percolation time.
The independent pilot determines the collision threshold through
`kappa=s*vw*max(pilot,tiny)**q_k`. The remaining graph is:

```
r = r0 + r1*vw
z = max(C0*finf/fanh*max(tau,tiny)**a_exp,tiny)
xi_dm = max(xi_floor**r + z**r,power_floor)**(1/r)
xi = max(xi_floor,xi_dm*max((1-f_bm)**lambda+A0*v_factor*g_bm,sum_floor))
```

The fixed quadrature supplies the bubble fraction f_bm and its inverse-radius
moment g_bm; v_factor accounts for wall velocity and expansion dilution. Its
parameter-independent radius distribution is cached. Powell handles the discrete
collision-radius threshold without assuming smooth derivatives.

`code/fit.py` uses up to 4,000 solver objective evaluations, checks physical
feasibility, and saves the best feasible training iterate. `PHY13_FIT_MAXFEV` can
set a smaller development iteration count. Baseline training and prediction use
NumPy/SciPy on CPU. The task allocation is 16 CPUs and 16 GB. Measured training
took 153.48 seconds, and validation prediction took 37.28 seconds.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: fit from the training file and write
  `model.json` plus a diagnostic `fit_report.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write a JSON
  certificate and ordered predictions to exactly FILE, regardless of extension.
- `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_SEED`, `BENCH_CPUS` and
  `BENCH_CODE_DIR` are provided by the runner.
- `bench/io_spec.md` specifies the editable expression graph and fixed numerical
  primitives. The scorer never imports candidate Python or unpickles a model.
- Normal commands are `bench eval-dev --seed 1`, `bench validate-output FILE`
  and `bench submit --final`.
- Dependencies are the two exact pins in `code/env.lock`.

## 6. Research scope

**Objective.** Improve prediction of the original lattice enhancement xi on held-out
physical slices, measured by natural-log RMSE.

**Fixed research objects.** The supplied lattice population, enhancement xi,
physical inputs, materialized slice protocol, seven-global-constant control,
independent auxiliary calibration, log-RMSE and physical checks. These keep the
prediction target, fitted capacity and physical extrapolation tests comparable.

**Editable methods.** Under `code/`, you may change the optimizer, initialization,
fitting objective, numerical approximations used during fitting, and enhancement
expression graph within the certificate's scalar arithmetic/physical primitive
vocabulary. The baseline expression is not mandatory. All target-fitted numeric
degrees of freedom must be represented by at most seven globally shared scalars.
Expression changes must remain compact physical models, not encoded tables or
extra fitted constants disguised as arithmetic or structure.

**Forbidden substitutions.** `data/`, `bench/`, this README, script signatures and
the certificate format are frozen. Do not modify or refit pilots, geometry or
auxiliary physics; import released global fits; hardcode targets; read unprovided
labels; or replace xi with a different endpoint. When retaining the supplied Powell fitting routine, changing only maxfev=4000 (PHY13_FIT_MAXFEV), maxiter=1000, xtol=1e-4 or ftol=1e-7 does not count as a method improvement. Changing only the `INITIAL` or `BOUNDS` constants in `baseline.py` also does not count. Submit a valid patch within the
overall budget. The deterministic baseline records the requested seed; randomized
alternatives must honor it. Training and prediction have no network access.

## 7. Known pitfalls

Do not use tiny finite differences for the original discrete geometry threshold.
Keep scalar certificates input-independent: row IDs and target values are not
available to expressions. The low-H physical checks extrapolate to 1e-6 and 1e-5;
passing only the lattice grid is insufficient.

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
