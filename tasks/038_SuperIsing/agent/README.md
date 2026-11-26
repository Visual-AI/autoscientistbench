# Super-Ising spectrum parameter optimization

## 1. Goal
Optimize h, mu_1 and mu_b in the fixed Super-Ising fuzzy-sphere Hamiltonian. The aim is to make its low-energy spectrum agree with the public conformal-bootstrap dimensions, measured by relative RMS error over 14 operators at N_mf=9. The model contains reference and excitation fermion flavours and a boson flavour. h controls scalar polarization; mu_1 and mu_b control the excitation-density terms. The physical model and t=1.5,U=.25,g=1 remain fixed.

## 2. Data
This is a fixed numerical problem with no statistical training split. data/train/problem.json states the model; data/val/input.json selects the full protocol. Optimization uses the supplied physical specification and public bootstrap dimensions. Validation is only for evaluation and method, hyperparameter or checkpoint selection, not for fitting reusable models or learned preprocessing. No archived spectra or results are inputs.

## 3. Metric and scoring
Cost (lower is better) is sqrt(mean((Delta/Delta_ref-1)^2)) over 14 operators. The common spectral scale is chosen analytically to minimize this cost. Frozen evaluation independently computes spectra from each submitted parameter case and returns the best N9 cost. The fixed reference list and energy-order assignment are in bench/eval_cost.py. The validator checks fixed couplings, five calibrations and at most 40 local evaluations. The spectral checks require residuals below 1e-6, spin errors below 1e-5, 40 states with 20 per sector, a spin-0 ground state and all 14 assignments. Invalid schedules are rejected.

The stated protocol checks do not certify unspecified external constraints. Any optimization procedure may propose the evaluated physical parameters. Measured validation cost is 0.03014293706878977, with standard deviation 0.0 from one reference run. PASS requires a relative improvement of at least 0.5% (delta=0.005), or the noise threshold if larger.

## 4. Baseline
The baseline uses FuzzifiED commit 3c84750fa9c4f439f6e3c77e7f80fda2288f0f45, persistent ED workers for each lz2 sector and SciPy Nelder-Mead. It retains five calibration evaluations, adaptive=False, the default 5% simplex and xatol=fatol=1e-4. Baseline local search uses a fixed 20 ED evaluations. This is editable baseline search logic, subject to the limit under Forbidden substitutions. ObsMomIncr=false remains. On 32 CPUs, measured optimization took 404.83 seconds. Validation parameter export took 0.07 seconds; independent validation ED scoring took 2315.95 seconds.

## 5. Interface
bash code/train.sh --out DIR runs optimization and saves the schedule. bash code/predict.sh --model DIR --input PATH --output FILE exports parameters; see bench/io_spec.md. Honor BENCH_SEED/BENCH_CPUS. Use bench eval-dev for validation evaluation with the frozen objective in bench/scorer.py. Julia 1.11.5, dependencies and Python/SciPy are supplied; env.lock is immutable in snapshot mode.

## 6. Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Optimize h, mu_1 and mu_b of the specified Super-Ising Hamiltonian for the frozen 14-operator relative RMS spectral cost at N_mf=9.

**Fixed research objects:**

- Original interacting Hamiltonian, t=1.5/U=.25/g=1, required observable convention and N_mf=9 physical problem.
- Same 14 bootstrap references, state/operator assignment, analytic optimal lambda, eigenpair/spin validity and RMS formula.
- Existing five-calibration/at-most-40-local-case protocol; frozen ED independently recomputes parameter cases. Unspecified external constraints are not certified.

**Editable methods:**

- Optimizer/search, initialization, numerical surrogate, parameter proposals and implementation for the same three-variable physical objective.
- Candidate numerical computation may change but cannot replace the frozen Hamiltonian or evaluation; Nelder-Mead and its stopping heuristics are baseline choices, subject to the limit under Forbidden substitutions.

**Forbidden substitutions:**

- Change Hamiltonian family, fixed couplings, physical size, bootstrap target, operator selection or cost normalization.
- Submit spectra/scores instead of parameter cases, omit required calibration or certify undefined constraints as passed.
- Replace full N9 with N4 smoke or use archived solutions instead of actual optimization.
- When retaining the supplied Nelder-Mead search, changing only adaptive=False, the default 5% initial simplex or maxfev=20 does not count as a method improvement. Hard-coding (h, mu_1, mu_b) values found during development, as a search start or as a submitted case, does not count as a method improvement.

Code is editable; physical model, frozen bench/,I/O and problem are fixed. No network or original reproduction access. The allocation is 32 CPUs and 144 GB memory. Scoring recomputes every supplied ED case. Submit parameters rather than fabricated spectra; candidate changes cannot modify the frozen Hamiltonian.

## 7. Known pitfalls
Julia uses a readonly depot with compiled caches disabled; the project resolves FuzzifiED to current candidate source. Frozen evaluation has separate readonly source. JIT overhead is visible in smoke runs. N4 smoke gets cost 1 because it omits the full N9 protocol. Full optimization and independent scoring both require substantial CPU time.

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
