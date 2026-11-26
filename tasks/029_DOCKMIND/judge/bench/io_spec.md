# Bounded six-parameter fit contract

The fixed public table is `data/train/train.csv`, with columns `target,pprop,hit_rate_mean,hit_rate_low,hit_rate_high,definition`. Prediction input has exact header `id,run,target,pprop,definition`; runs0..4 each contain the complete fixed instance. Supplied observations and hit definitions are immutable.

Prediction header must be exactly `id,pred,rho,exp_mean,exp_std,artifact_freq,artifact_mean,artifact_std`. Return every input ID in order. For each target/run, all rows must carry one identical finite six-parameter vector. Bounds are rho[-0.9,-0.4], exp_mean[-4,4], exp_std[0.5,3], artifact_freq[1e-7,1e-2], artifact_mean[-5,-3], artifact_std[0.5,1].

The fixed bivariate-normal plus Gaussian-artifact model is interpreted by trusted `hit_rate_model.HitRateModel`. Its frozen inverse-CDF lookup is part of the numerical evaluation target. Claimed `pred` must match trusted recomputation within rtol1e-10/atol1e-12. The two-column free-prediction interface is not accepted. MSE is calculated from recomputed values, averaged within each target, equally across the three targets and then across five runs.

Any fitting/search/optimization algorithm may find feasible parameters; the baseline DEAP implementation is not mandatory. Randomized fit r uses literal seed r; BENCH_SEED chooses execution order. Deterministic fits may legitimately agree. Do not copy a best random restart into other runs or use published fitted answers.

The full 93-point table is supplied in `data/train/train.csv`. Fit only to this designated training table. The public validation files repeat its observations for evaluation and selection; do not fit models or learned preprocessing on validation. This fixed-instance parameter optimization task does not measure generalization to held-out molecules. Direct target lookup and interpolation remain forbidden. Independent fitting provenance requires source audit; the scorer enforces the model, bounds, complete groups and unchanged MSE formula directly.

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
