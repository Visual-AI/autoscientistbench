# A Residence-Time Approach for Determining Position-Dependent Diffusivities from Biased Molecular Simulations

## 1. Goal

Estimate a position-dependent water diffusivity profile from biased molecular-simulation first-exit statistics. The evaluator uses the given ABF potential of mean force (PMF) to propagate that design with the periodic Smoluchowski equation, starting at the membrane centre, and compares it with unbiased molecular dynamics. Improve `tv` (lower is better) on reference densities held by the judge. POPC is the public validation system. You may improve the method under the rules in §6.

## 2. Data

- `data/train/` contains ABF statistics and PMFs for the supplied physical systems. Each directory contains observed first-exit segment lengths/counts and their cell keys (`.npy`); `statistics.npz` gives interval widths, positions, origin counts, blocking SEMs and censoring fractions. `meta.json` gives sampling walls and physical parameters. `context.npz` describes the propagation grid. The supplied PMF text is in kcal/mol; diffusivity units are Å²/ps.
- `data/val/input.npz` and `data/val/labels.npz` describe POPC at 310 K. Labels contain its unbiased-MD densities on 763 grid points at 9 available lag times. This is a fixed numerical problem, without a random row split or supervised training targets.
- Fit profiles and learned preprocessing only on the designated training observations. Use validation features and MD densities only for evaluation and method, hyperparameter or checkpoint selection.
- Raw trajectories are not redistributed by the authors. All RTA estimates are recomputed from the released histograms. Time-order-dependent SEM metadata is retained from the cache. Fitted D/NLL and published solutions are not provided.
- Do not re-split, relabel or obtain additional data from outside sources.

## 3. Metric and scoring

The frozen scorer solves the Smoluchowski equation from your submitted D. Total variation is half the grid-spacing-weighted absolute difference between the normalized predicted and reference densities. `tv` is mean TV at 75 and 200 ps if the mean over all reference lags is at most 0.0435 on test or 0.18 on val; otherwise `tv=1`. It always uses the fixed ABF PMF, periodic boundaries and central initial condition. Invalid designs receive Infinity. Diagnostics include `tv_raw` (the unpenalized headline), `tv_all`, `quality_guard_ok`, and `tv_by_lag`.

| Guard | Enforcement |
|---|---|
| Mean TV across supplied lags ≤ 0.0435 on test, ≤ 0.18 on val | scorer; otherwise 1 |
| Fixed ABF PMF, grid, periodic propagation and initial condition | scorer; trusted physical inputs and solver, no candidate override |
| Exact identifier set, finite positive diffusivity, strict CSV format | validator and scorer; invalid output rejected |
| Compute from permitted statistics; no memorized/precomputed solutions or hidden MD access | patch audit |

Measured public validation baseline: `tv=0.10001307621145611`, `tv_raw=0.10001307621145611`, and `tv_all=0.17010629542906788`. The baseline passes the validation quality guard. The formal-score standard deviation is 0.0 from one reference run. Values and timings are also reported by `bench status`. The judge compares its mean score with the archived baseline. PASS requires relative improvement of at least 0.001, subject to the variance rule.

## 4. Baseline

Official code: https://github.com/mvondomaros-lab/pdda at `805cddfae1930239abaf63b1b94d9f669333282a`. The RTA baseline treats every sampled frame inside an interval as a time origin and reconstructs its mean remaining first-exit time, tau, from the histograms. It estimates `D = L²/(12 tau)` for interval width L. It evaluates interval-width and NLL diagnostics, then uses L=7.5 Å. Positive, symmetric smoothing uses length 1 Å, plateau tolerance 0.05 and extrapolation. Scientific code is in `code/pdda/`; reconstruction is in `code/rta_common.py`. The resulting profile is compared with unbiased transport through the same fixed physical forward model.

Training reconstructs and saves diffusivity profiles from the designated ABF training statistics. Prediction loads the corresponding fitted profile and writes the required design output. Physical propagation runs in the frozen scorer.

On 16 CPUs, measured training with one single-threaded process per CPU over the cells took 60.33 s; validation prediction took 1.97 s. The task reserves 16 GB memory; this is an allocation, not a measured peak. Scoring performs the separate frozen physical simulation.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` reconstructs the diffusivity profiles and writes all model artifacts into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with exactly `id,diffusivity`. Full format: `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`. Paths passed by the harness are absolute.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, `bench submit --final` are the development interfaces.
- Dependencies are pinned in `code/env.lock` for Python 3.12. PyPI `name==version` pins are rebuildable; editable, Git and local-file installs are not. `bench submit` freezes the environment.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


Diffusivity estimation, regularization and optimization under `code/` may change within the fixed physical research scope below, while retaining script arguments and output format. `data/`, `bench/`, this README and the physical evaluation protocol are frozen. The guard table in §3 states which rules the scorer checks and which require patch audit; there is no additional candidate-side certificate or unadvertised probe.

Compute designs from the permitted ABF statistics. Do not embed precomputed solutions, hard-code profiles by system/id, digitize paper curves into predictions, read or infer withheld MD data, or override the PMF and physical evaluator. Candidate predictions are designs; a density table or self-reported score is not a valid submission.

The allocation is 16 CPUs and 16 GB memory. The judge uses one seed. Honor `BENCH_SEED` if introducing randomness. Train/predict have no network access.

## 7. Known pitfalls

POPC's released MD reference uses the smaller periodic box, while its ABF statistics come from the larger trajectory described by the source repository. Interpret its raw diagnostics together with the validation quality guard.

The frozen scorer performs numerical propagation, so scoring itself takes time after predict finishes. Large diffusivities can make its sparse exponential more expensive. Do not normalize or rescale D as if it were a probability density. The grid is the author's odd-sized periodic grid near 0.1 Å spacing, not an independently constructed linspace. Output exact supplied IDs with no duplicates. Setting BLAS thread counts beyond `BENCH_CPUS` is unsupported.

## Research scope

**Objective.** Infer a positive position-dependent diffusivity profile from the released biased-simulation first-exit observations, minimizing the original propagator total-variation objective.

**Fixed research objects.** existing physical systems and ABF first-exit observations, PMF, temperature, periodic grid, boundaries and central initial condition; Smoluchowski physical forward model, reference MD densities, headline 75/200 ps lags, all-lag quality cap and original TV aggregation.

**Editable methods.** diffusivity estimators, statistical fitting and regularization, new interval-selection and smoothing procedures computed from authorized observations; optimization/training surrogate losses and computational implementation without changing the physical evaluator.

**Forbidden substitutions.** replacing the Smoluchowski equation, PMF, initial/boundary conditions or reference densities with an easier physical model; submitting arbitrary densities or cached profiles instead of a freshly inferred positive diffusivity design. When retaining the supplied RTA reconstruction and smoothing pipeline, changing only L=7.5 Å (including switching to the supplied NLL width selection), smoothing_length=1.0, plateau=True, plateau_tolerance=0.05, symmetry_center=0.0 or positive=True does not count as a method improvement.

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
