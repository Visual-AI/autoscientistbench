# Manifold-Aware Perturbations for Constrained Generative Modeling

## 1. Goal

**Research question.** Data under an equality constraint lie on a lower-dimensional manifold, so their distribution has no density in the ambient space. A diffusion model trained on them meets score explosion near the manifold, with unstable sampling and an inaccurate generated distribution. The paper asks whether a cheap change of target avoids this: perturb each point only along the local normal directions to get a lifted distribution `p_sigma`, train an ordinary unconstrained model on it, then project the samples back onto the manifold.

**Experiment.** Paper Section 5.3 and Table 2 test this on the Stanford Bunny surface, which has greater and more irregular local curvature than the paper's plane and sphere. At sigma 0.0005, Table 2 compares five diffusion methods on one target: `p_sigma` with projection (the baseline here), DDPM on the unperturbed data, that DDPM with its samples projected afterwards, PDM (projection during sampling) and PIDM (a constraint residual loss during training). It reports COV, JSD and TVD. The paper keeps the model and training hyperparameters the same across these methods, so differences come from how each handles the constraint. It sweeps sigma in a separate figure. In Table 2 the three methods whose samples end on the mesh reach similar COV, well above plain DDPM and PIDM, which produce off-manifold samples.

**Your job.** Learn the heat-kernel distribution on the Bunny surface. The baseline learns its normally perturbed distribution with a lifted DDPM at sigma 0.0005; samples are then projected onto the mesh. Improve extrinsic `cov` (higher is better) against reference points held by the judge. You may improve the DDPM under the rules below. The method must stay a DDPM: Table 2 compares diffusion methods, and this task asks how far such a method can go on this target.

## 2. Data
- `data/train/samples.npy`: 100000 float32 lifted points at sigma 0.0005, three normalized coordinates, generated with public seed 42. `data/train/mesh.obj` is the public Stanford Bunny geometry, with 35947 vertices and 69451 faces.
- `data/val/input.npz`: 10000 opaque sample-slot IDs. `data/val/truth.npz`: public reference coordinates and frozen scoring metadata; reference seed 20261029. Reference points are unlifted mesh vertices.
- `bench/generator.py` is the frozen author data generator. It samples mesh vertices with heat weights computed from Euclidean distance, and optionally adds noise along vertex normals. The paper describes geodesic distances and the function name says geodesic, but the implementation is Euclidean. This behavior is intentional.
- Requests contain only opaque IDs; no reference seed, coordinates or vertex identity is provided.
- Repeated vertex coordinates are natural in this discrete distribution. Do not re-split or relabel the data, fetch outside data, or generate additional training/reference datasets. Validation labels are for evaluation and method, hyperparameter or checkpoint selection only, not a prediction lookup table.

## 3. Metric and scoring
The frozen scorer computes `coverage(projected_generated, reference, k=5)`, exactly the author's Bunny plotting convention. For each finite projected generated point, the radius is its fifth-neighbor distance within the generated cloud, including itself. It is covered if the closest reference point is within that radius. COV is the covered fraction of generated points, with upper bound 1. The paper's Appendix D writes coverage with balls centered on real samples; this task keeps the plotting order, with balls centered on generated points. COV therefore does not directly penalize missing parts of the target. The JSD and TVD guards below hold the generated distribution to the reference.

| Guard | Requirement | Enforcement |
|---|---|---|
| Output format | Exactly `id` and `samples`, the exact ID set with no duplicates, floating samples of shape `(n, 3)` | validator |
| Numerical failures | At most ten nonfinite raw rows and at most ten nonfinite projected rows | scorer; validator checks raw format |
| Failure handling | Do not drop, replace or resample failed rows | patch audit |
| Mesh projection | Scoring always uses the frozen author's MeshConstraintProjector output | scorer |
| Distribution fidelity | Fixed-range JSD and TVD must each be no greater than their same-reference baseline mean plus twice the sample SD | scorer |
| Information restriction | Never read reference labels while predicting, hard-code outputs, or infer hidden seeds | patch audit |
| DDPM method | Every output row comes from reverse diffusion of the trained denoiser, with no snapping or reassignment to mesh vertices or training points | patch audit |

Histograms have 50 bins per coordinate, bounds fixed to the reference bounding box plus 2% of each coordinate span on both ends, one overflow bin, count smoothing 1e-12, and natural logarithms. Nonfinite and out-of-range samples count in overflow. Coverage uses finite projected samples. The paper also discards NaN/Inf samples before computing metrics, usually no more than 10 out of about 10^4. Guard violations return `Infinity`, which the harness treats as invalid. The validation guard means and SDs are frozen from three full baseline runs on the validation reference.

On validation the caps are JSD 0.1504 and TVD 0.3526. The three reference baseline runs measured JSD 0.1460 to 0.1478, TVD 0.3453 to 0.3526, and 0 to 3 nonfinite rows, so the distribution caps leave little room above the baseline.

Measured validation baseline: **0.8566765672**, seed standard deviation **0.0110070086** (three full retrainings). `bench status` exposes the baseline. The judge retrains from scratch with three seeds; PASS requires relative improvement at least `max(0.02, 2 * baseline_seed_sd / baseline_mean)` and every seed passing its guards.

## 4. Baseline
Official code: https://github.com/katiekeegan/ManifoldAwarePerturbationsForConstrainedGenerativeModeling at commit `376244fe81f25fcc5e8cacd9908079d27d4f026a`. `bench_run.py` is the entry point behind both scripts, `models.py` defines the denoiser, `trainers.py` the unchanged DDPM training and sampling routines, and `utils/constraints.py` the mesh projector. `DDPMTrainer` keeps the authors' switches for the PDM sampler and the PIDM loss. `models.py` and `trainers.py` also hold the authors' normalizing-flow code, which the rules below do not allow as a replacement.

The baseline uses 100000 training samples, 200 epochs, hidden width 128 (17539 parameters), batch size 64, Adam learning rate 1e-3, T=250, the original linear beta schedule and time concatenation. It saves the final model without selecting on evaluation data. Algorithm randomness is separate from frozen training-data generation. The baseline retains the author models, loss, optimizer, training loop and sampler, including the sampler's numerical bugs.

The task provides 16 CPUs and a 16 GB memory budget. The baseline uses up to eight numerical threads, matching its source configuration. Measured training averages **7.30 min** and validation prediction **0.23 min** at this CPU allocation. Memory is a configured budget, not a measured peak.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` trains and writes all required model artifacts to DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes NPZ with exactly `id` and `samples`; see `bench/io_spec.md` for shapes, coordinates and failure handling.
- Environment: `BENCH_SEED` controls algorithm randomness; `BENCH_CPUS` limits threads; `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR` identify the data, frozen helpers and code directories.
- `bench eval-dev --seed 1` runs one algorithm seed; `bench validate-output FILE` checks an output file.
- Only PyPI/PyTorch CPU `name==version` pins are rebuildable; editable, git and local installations are unsupported.

## 6. Rules
- Editable methods and forbidden substitutions are in the Research contract below. You must keep learning the same surface distribution from the supplied lifted cloud and preserve the frozen projection/metric. The geometric problem and input data are fixed.
- Frozen surface: `data/`, `bench/`, this README, the prediction format and script signatures.
- Every run must honor `BENCH_SEED` and the CPU budget. Seed variance contributes to the improvement threshold.
- No network during training or prediction; required dependencies must be reproducible from the lock.
- Do not read or guess the hidden split, use reference labels as samples, hard-code outputs, embed remembered paper results, or bypass the frozen scoring interface. Public mesh geometry may be used for geometric computations, but the generator must not be used to draw extra reference/training sets or implement an exact-distribution oracle. These information restrictions require patch review.

## 7. Known pitfalls
Keep the Bunny plotting argument order: reversing COV arguments changes the metric. The sampler already projects at its last step; the scorer projects its submitted output again, as the author evaluator does. Preserve failed raw rows and their IDs; do not silently drop or replace them. A valid output may have up to ten nonfinite rows. Save NPZ through a binary file handle because the requested filename may have no extension. Absolute paths are passed to entry points, and `PYTHONSAFEPATH=1` requires an explicit package path. The baseline is stochastic even though a fixed seed is repeatable.

## Research contract

**Objective:** Learn the fixed Bunny surface distribution from the supplied lifted training cloud and public mesh, maximizing the frozen projected-sample COV under distribution-fidelity guards.

**Fixed research objects:** The original mesh/coordinate system, 100000 lifted training points at sigma 0.0005 and the fixed heat-weight target distribution with author Euclidean-distance behavior remain fixed, because Table 2 compares methods on one target. The reference distribution, raw 3D output semantics, authoritative author mesh projection, generated-centered COV k=5, fixed JSD/TVD histogram formula and baseline-relative guards remain fixed, so baseline and candidates are measured the same way (scorer). At most ten nonfinite raw/projected rows (scorer) and original failed-row handling (patch audit) remain fixed.

**Editable methods:** Within DDPM, the denoiser architecture, auxiliary learning losses, internal feature representations, noise schedule, perturbation design, optimizers, training procedure and reverse-diffusion sampler may change, including constraint-aware variants such as projection or physics-informed terms (PDM and PIDM in the paper). The paper holds model and training settings constant across its methods; here they are editable. Public mesh geometry may support geometric computations inside training and the reverse process. Every variant must still learn from the same released training sample rather than substitute another target or data source.

**Forbidden substitutions:** The denoiser must be a neural network fitted by gradient training; it must not store training points or per-vertex weights or counts, and must not compute a closed-form posterior over training points or mesh vertices. Do not change the surface/target weights/noise data, replace projection or reverse COV arguments, drop/resample failed rows, or replace distribution learning with a fixed-reference lookup. Do not replace the DDPM with another generative family or a non-diffusion estimator, such as a flow, GAN, VAE, vertex histogram, kernel density or nearest-neighbor lookup of training points. Do not post-process samples by snapping or reassigning them to mesh vertices or training points, by jittering such points, or by resampling or reweighting them. No additional generated training/reference sets, exact-distribution oracle from the frozen generator, external data/pretraining or reference-label fitting. No training or fitting of reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. The scorer does not detect most of these, so they are checked by patch audit; a valid score does not make a substitution acceptable.

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
