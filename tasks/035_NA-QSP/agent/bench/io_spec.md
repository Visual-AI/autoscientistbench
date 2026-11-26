# Pulse design interface

Input and labels are CSV with header `id,alpha,theta_t`; alpha is positive real coherent amplitude, theta_t is in radians. Labels repeat the instance parameters so the scorer can simulate independently. Validation contains one public instance. Delta=1 and the CD budget is always pi/4.

Output is CSV with header exactly `id,pulse,n_gates`. One row per input id, no duplicates or omissions, arbitrary row order. `pulse` is a JSON list serialized and CSV-quoted in a single line; `n_gates` is its integer length (zero is allowed). All parameters must be finite real numbers. No other columns or gate fields are accepted by the scorer.

Gates act in listed execution order:

- `{"type":"cd","lam":0.1,"phi":0.0,"qubit_axis":[1,0,0]}` implements exp(-i lam (cos(phi) x + sin(phi) p) sigma_n).
- `{"type":"qubit_rot","angle":0.2,"axis":[0,0,1]}` implements exp(-i angle sigma_n/2), at no CD cost.

Axes are real unit vectors within 1e-12 and normalized within that tolerance. The total CD cost is math.fsum(abs(lam)); it must be at most pi/4 without an added tolerance. Angles use radians. Qubit rotations are free; there is no fixed gate count.

The frozen simulator uses oscillator-first tensor order, x=(a+a.dagger)/2 and p=-i(a-a.dagger)/2. Inputs are |+alpha> tensor |g>, |-alpha> tensor |g>, and the normalized even cat tensor |g>. Targets are |+alpha> tensor rx(theta_t)|g>, |-alpha> tensor rx(-theta_t)|g>, and their normalized coherent sum. This preserves the reproduction's approximate branch-target convention, including finite coherent-state overlap; the target is not constructed by exponentiating sign(x).

For each input, 1-FH=1-abs(target.overlap(normalized_output))**2 at Ncav=400. Each instance uses the maximum over the three inputs. The task metric `hybrid_infidelity` is their mean over supplied instances. Every input's infidelity must agree at Ncav=600 within absolute 1e-9. Simulated state vectors must preserve their norm within absolute 1e-10. The physical worst score is 1.0; guard violations report `hybrid_infidelity: Infinity` as the harness rejection sentinel and `score_rule_value: 1.0`. Diagnostics include individual infidelities and cutoff differences; submitted numeric scores or states are never trusted.

`train.sh --out DIR` saves config.json. `predict.sh --model DIR --input PATH --output FILE` reads that configuration and produces designs for the supplied instances. The output path need not have a file extension. Frozen helpers are importable by absolute path from BENCH_BENCH_DIR; they do not depend on code modules or external files.

## Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Minimize hybrid infidelity of the specified sign-controlled qubit rotation under the fixed conditional-displacement amplitude budget.

**Fixed research objects and controls:**

- Supplied coherent amplitude/angle; oscillator/qubit conventions, both coherent branches, even-cat input and prescribed targets.
- CD and unconditional qubit-rotation gate set, unit axes, total absolute CD amplitude <=pi/4 and Delta=1.
- 400/600 cutoff convergence test, worst-of-three-input infidelity and mean across instances.

**Editable components:**

- Pulse sequence, gate count, allowed axes/angles/amplitudes within the fixed gate set and amplitude budget.
- Analytical designs, optimization, reusable learned design policies, search and implementation; two-gate GCR is a baseline.

**Forbidden substitutions:**

- Change target transformation, coherent/cat inputs, physical simulator, gate set or amplitude budget.
- Replace fixed cutoff/fidelity with an easier evaluation or submit fabricated states/scores instead of pulses.
- Use stored per-instance solutions, withheld-collection access or ID-dependent answers.

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
