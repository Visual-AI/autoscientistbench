# Tera molecular energy prediction under a CPU resource budget

The judge retrains submitted code from scratch and compares hidden evaluation
scores against an archived reference.

## 1. Goal

Research question: how can a Gaussian process (GP) use observed gradients to
predict function values when both the sample count n and the input dimension d
are large? Exact inference with n values and n full gradients has a joint state
of size n(d+1) and scales cubically in it. The paper "Scalable Derivative Gaussian
Processes via Exact Gradient Reduction" proposes Tera. Tera conditions each target
on m neighboring observations (a Vecchia approximation). For stationary kernels,
the target's conditional density depends on those md gradient coordinates only
through at most m^2 target-specific directional derivatives, so each factor costs
O(dm^2 + m^6) time.

The experiment is large-scale regression on six MD22 molecular dynamics
trajectories, with d from 168 to 1110. Inputs are Cartesian atomic coordinates,
the target is held-out molecular energy, and observed forces are conditioning
information, not a scored target. The paper compares Tera with a function-only exact GP
and two derivative GP approximations (DDSVGP, DSoftKI) on identical splits, by raw
energy RMSE per atom over three independent runs, under a limited resource budget.

Your reference is the paper's isotropic squared-exponential (SE) Tera configuration.
Minimize `energy_rmse_sum`, the sum of six three-replica mean raw energy RMSEs per
atom, under the resource limits below. You may improve the method within the frozen
interface and rules. Tera is one answer to the question; the model family, a GP
conditioned on observed energies and forces, is fixed.

## 2. Data

`data/train/d0.npz` through `d5.npz` contain original frame IDs, float32 coordinates
R (Angstrom), molecular energies E (kcal/mol), forces F (kcal/mol/Angstrom, the
negative gradient of E with respect to R), and atom counts. Dataset order is DHA,
AT-AT, stachyose, AT-AT-CG-CG, buckyball-catcher, double-walled-nanotube, with 56,
60, 87, 118, 148 and 370 atoms. Training / validation frame counts by dataset are
40726/10182, 11689/2923, 15879/3970, 5906/1477, 3557/890, and 2934/734.
`bench/io_spec.md` defines all shapes.

`data/val/input.npz` and `data/val/labels.csv` are public validation data. They list
the same frames once per replica. Validation is the final 20% by trajectory order of
the eligible training frames; this is a temporal holdout, so its error may differ
from randomly held-out data. The paper has no validation split. All public files
exclude evaluation frames. The evaluation partitions are held by the judge and are
never accessible here. Three replicas are evaluated separately, without averaging
their predictions.

Both baseline and candidates fit only the training files, including final fitting.
Validation may be used for model selection only. External data, downloads,
and reconstruction of unavailable trajectory frames are forbidden.

## 3. Metric and scoring

`energy_rmse_sum` is lower-is-better, in kcal/mol/atom. For each dataset and
replica compute `sqrt(mean((predicted_energy - energy)^2)) / n_atoms`, average
its three replica scores, then sum the six dataset means. The scorer prints
per-dataset and per-replica diagnostics. This is neither a pooled RMSE nor an
ensemble-prediction RMSE.

| Guard | Enforcement class |
| --- | --- |
| At most 64 allocated CPUs | scorer checks resource record; independent judge probe |
| Peak process-tree memory at most 56 decimal GB | scorer checks resource record; independent judge probe or patch audit |
| Output IDs match the input once each, all numbers finite, resource fields identical on every row | validator and scorer |
| Resource record includes all training and prediction work, and is unmodified | judge probe or patch audit |
| Predictor remains a GP conditioned on observed energies and forces | patch audit |
| All fitting in `train.sh`, on the training split only | patch audit |
| No external data, hidden-frame reconstruction, or memorized outputs | patch audit |

Any resource violation or invalid prediction file gives `Infinity`. Frozen
`bench/resources.py` must monitor all work, including loading, imports, scheduling,
fitting and saving. Resource records are checked by independent judge measurement
or patch audit; a self-reported certificate alone is not proof. The scorer sets no
limit on `train_s`. Time is limited by your overall deadline and by the judge's
overall replay budget, not by a per-script cap.

Measured validation reference: **1.4328999855** kcal/mol/atom
(population standard deviation 0.0061892691, three algorithm seeds).
`bench status` also reports the baseline validation score.
PASS uses relative improvement at least `max(delta, 2 * reference_std / reference_mean)`;
delta is 3%, and the judge uses three algorithm seeds.

## 4. Baseline

Official code: https://github.com/hseung88/tera at
`613493b8b48e9982a4461f92e99e9fe9daa6a618`. Numerical model, likelihood,
optimizer, graph ordering and batched predictor are in `code/src/tera/methods/`
(`tera/` and `common/`). The baseline uses an isotropic SE kernel, m=20, x_scale=3,
float32, one Adam epoch at learning rate 0.01, training batch 256 and prediction
batch 32. Kernel and both noise parameters are learned on the conditional Vecchia
likelihood, using the authors' initialization and clipping.

The paper uses the isotropic SE kernel in this comparison because the DDSVGP
implementation supports neither ARD nor Matern kernels. It also reports Tera with
an ARD Matern-5/2 kernel. Its sweep of m in its own regression experiment improved
up to m=20, with no further gain at m=30. The O(m^6) solve makes large m costly.

The baseline standardizes energy using training data and scales force gradients
by the chain rule. Each run fits six datasets times three replicas, scheduling at
most twelve independent four-thread processes. Model selection does not inspect
evaluation inputs or labels. Measured on 64 CPUs: training 11.82 minutes on average
(range 11.34-12.18); public validation prediction averages 0.89 minutes. Maximum
reference process-tree memory was 26.39 GB.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` trains and saves all predictor artifacts plus `resources.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes the exact output path as CSV.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`; absolute paths are passed by the harness.
- `bench validate-output FILE` checks a prediction file.
- Dependencies are pinned in `code/env.lock`. Install with uv; only exact PyPI/PyTorch CPU package pins are rebuildable.

## 6. Rules

The predictor must remain a Gaussian process conditioned on observed energies and forces. Its kernel, derived features, conditioning structure, training surrogate loss
and optimization under `code/` may change within the research contract below,
preserving the observed-energy target, output format and resource limits. The scorer
does not check the GP rule; patch audit does. Frozen surfaces are `data/`, `bench/`, this README and script signatures.
Final training must use the training split only; validation is for evaluation and selection only. Do not use external data,
hard-code predictions, reconstruct hidden frames or embed paper result tables.

The judge runs three seeds at 64 CPUs. Honor `BENCH_SEED` in every
randomized algorithm. The memory limit is 56 GB. No network is available during training or prediction.

## 7. Known pitfalls

The target is whole-molecule raw energy; divide RMSE by atom count only inside
the scorer. The baseline internally models negative standardized energy and
rescales force gradients by x_scale: `x = R/3`, `y = -(E - mean)/std`, `g = 3F/std`,
and predictions convert back by `E = mean - std * y`. Preserve this convention when
using its core. Float32 rounding of large absolute energies matters, and changed BLAS
threading can change ill-conditioned solves. The baseline uses four threads per worker.

Keep `PYTHONPATH` explicit because the harness sets `PYTHONSAFEPATH=1`. Data and
benchmark helpers are read-only. All model artifacts must be saved in `--out`;
only committed code enters the clean retraining environment. Predictions are CSV,
including finite resource columns identical across rows, even when the requested
output filename has no extension. Resource records cannot be fabricated or edited.

`code/env.lock` pins torch, numpy and torch's dependencies. `main.py` does not use
the upstream modules for other methods and experiments; some, such as `ddsvgp`, import
packages that are not pinned.

The upstream training log may show `raw_rmse=nan` because evaluation callbacks
are disabled during training. This diagnostic is not the prediction output;
the validator requires every submitted energy and resource value to be finite.

## Research contract

**Objective.** Predict raw molecular energies for the six fixed MD22 trajectory datasets under the original CPU/memory budget, minimizing the sum of three-replica mean energy RMSEs per atom.

**Fixed research objects.**
- A Gaussian process conditioned on observed energies and forces, because the paper studies scalable derivative GPs.
- Original six molecular datasets, coordinates/energies/forces, physical units, atom counts and materialized splits, because they make scores comparable with the reference.
- Eighteen independent fitted predictors, all fitting in train, raw-energy targets and original per-replica/per-dataset RMSE aggregation, because the metric is defined over them.
- At most 64 allocated CPUs, peak process-tree memory<=56 decimal GB and complete resource monitoring, because accuracy at practical cost is the point of the experiment.

**Editable methods.**
- Within Gaussian processes conditioned on observed energies and forces: kernel, legal descriptors/features, conditioning structure, training surrogate loss and numerical optimizer.
- Internal scaling, approximation and scheduling within the same observed-energy prediction problem and resource budget.

**Forbidden substitutions.** Original frame numbers in IDs and `id` arrays are row keys only: do not use them as model inputs, to choose conditioning sets, or to infer which frames are held out. The GP prior mean must be fitted on the training split only and must not carry the prediction: no mean or energy offset may be selected on validation, and no neural network, tree ensemble or other non-GP regressor may serve as the mean.
- Replacing the Gaussian process with a neural network, tree ensemble or other non-GP predictor; replacing observed molecular energy with agreement to a chosen GP posterior, an easier molecular dataset or altered physical units.
- Pooling/ensembling away independent replica scores, fitting during prediction, modifying resource records or reconstructing unavailable frames.
- Training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
