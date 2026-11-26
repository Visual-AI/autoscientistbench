# Parameter-schedule interface

Input JSON identifies full or smoke mode. Full problem: fixed Super-Ising Hamiltonian, t=1.5,U=.25,g=1. Output has exactly mode,fixed_couplings,cases. Each case has integer nmf, unique alphanumeric tag, and three finite parameters [h,mu_1,mu_b].

Full mode requires original table6/table7/table8/table9/example9 calibrations in original order and with published triples (listed in contract.py), then zero to forty local01...local40 cases at N9. Frozen scorer independently runs all spectra and chooses best N9 cost; candidate energies, scores, residuals and filenames are never accepted. Analytical lambda and14-operator assignment are unchanged.

The protocol checks unchanged Hamiltonian/couplings; N9 spectrum and14-operator coverage; five calibrations, analytical lambda and at most40 local evaluations. Unspecified external constraints cannot be certified. The candidate may replace the optimizer; the physical objective and parameter-schedule format remain fixed.

Smoke uses one N4 case, runs actual candidate and frozen ED, and receives worst1 because full N9 protocol is deliberately omitted. Measured full validation results are reported in README.md.

## Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Optimize h, mu_1 and mu_b of the specified Super-Ising Hamiltonian for the frozen 14-operator relative RMS spectral cost at N_mf=9.

**Fixed research objects and controls:**

- Original interacting Hamiltonian, t=1.5/U=.25/g=1, required observable convention and N_mf=9 physical problem.
- Same 14 bootstrap references, state/operator assignment, analytic optimal lambda, eigenpair/spin validity and RMS formula.
- Existing five-calibration/at-most-40-local-case protocol; frozen ED independently recomputes parameter cases. Unspecified external constraints are not certified.

**Editable components:**

- Optimizer/search, initialization, numerical surrogate, parameter proposals and implementation for the same three-variable physical objective.
- Candidate numerical computation may change but cannot replace the frozen Hamiltonian or evaluation; Nelder-Mead and its stopping heuristics are baseline choices, subject to the limit under Forbidden substitutions.

**Forbidden substitutions:**

- Change Hamiltonian family, fixed couplings, physical size, bootstrap target, operator selection or cost normalization.
- Submit spectra/scores instead of parameter cases, omit required calibration or certify undefined constraints as passed.
- Replace full N9 with N4 smoke or use archived solutions instead of actual optimization.
- When retaining the supplied Nelder-Mead search, changing only adaptive=False, the default 5% initial simplex or maxfev=20 does not count as a method improvement. Hard-coding (h, mu_1, mu_b) values found during development, as a search start or as a submitted case, does not count as a method improvement.

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
