# Non-Abelian Quantum Signal Processing: composite pulse design

## 1. Goal
Design a pulse implementing a sign-controlled qubit rotation for an oscillator and qubit. Given coherent amplitude alpha and target angle theta_t, output conditional-displacement (CD) gates and free unconditional qubit rotations. Minimize `hybrid_infidelity` (lower is better), preserving both coherent branches and their even-cat superposition. Delta=1; the total CD amplitude budget is pi/4. Evaluation also uses other (alpha, theta_t) instances, supplied only through the prediction input. You may change the design method under the rules below.

## 2. Data
- `data/train/config.json` is the public GCR configuration. No training dataset is supplied. Validation is only for evaluation and method, hyperparameter or checkpoint selection, not for fitting reusable models or learned preprocessing. Per-instance pulse optimization may use that instance's supplied parameters and the public simulator, without carrying fitted state between independent validation cases.
- `data/val/input.csv` and `data/val/labels.csv` contain one public instance: alpha=2, theta_t=pi/2. Labels repeat the parameters required by the independent simulator.
- Each prediction input supplies the instance parameters to use. Instance ids carry no usable information.
- You may simulate additional development instances yourself and use the public simulator. Do not infer or reconstruct the held-out instance collection. No outside datasets are needed.

## 3. Metric and scoring
`bench/scorer.py` independently applies the submitted pulse using QuTiP at Ncav=400. For each input it computes the state-overlap infidelity, 1-FH=1-abs(target.overlap(normalized_output))**2. Each instance uses the maximum over coherent +alpha, coherent -alpha, and the normalized even cat; multiple instances are aggregated by their arithmetic mean. The theoretical lower bound is zero.

The targets are coherent +alpha tensor rx(theta_t)|g>, coherent -alpha tensor rx(-theta_t)|g>, and their normalized coherent sum. The objective is to preserve the oscillator branches and their coherence while rotating the qubit. These prescribed branch targets retain the approximation for overlapping coherent states; they are not obtained by exponentiating sign(x).

| Guard | Enforcement |
|---|---|
| Only CD and unconditional qubit rotations; finite parameters; unit axes within 1e-12 | scorer |
| Sum of absolute CD amplitudes at most pi/4, with no added tolerance | scorer |
| Each input's 400/600 infidelity difference at most 1e-9 | scorer |
| Simulated state-vector norm error at most 1e-10 | scorer |
| Infidelity recomputed from pulse; no submitted score or state is accepted | scorer |
| No lookup by id, stored instance solutions, or access to withheld collection | patch audit |

Physical guard failure has worst value 1.0. The harness receives `Infinity` as a disqualification sentinel, with `score_rule_value: 1.0` in diagnostics. Format errors fail validation. The measured validation reference is `0.002983169219097581` (sigma=0). Deterministic baseline: one seed. PASS requires relative improvement at least 0.001% (delta=1e-5), or the noise threshold if larger.

## 4. Baseline
The baseline is the analytical Gaussian-controlled rotation (GCR): apply the momentum CD about sigma_y, then the position CD about sigma_x, each with lambda=theta_t/(2 alpha). The momentum pre-correction suppresses the first-order rotation error caused by Gaussian position uncertainty. Candidate designs are compared with this baseline under the same physical inputs and CD budget. The official repository is https://github.com/shraggy/NA-QSP_sims at commit `298f15adf98fcb5973146fa0f818293a5827d5f7`. Its notebooks remain under `code/`.

The training entry point in `code/main.py` saves the public configuration without fitting. Prediction computes a pulse for each supplied instance and writes CSV. The frozen scorer independently simulates the submitted pulse. Auxiliary figure scans are not part of training or prediction.

On 8 CPUs, reference train and validation prediction each took 0.22 s. Prediction timing covers CSV pulse generation. The independent physical simulation runs in the scorer and is not included in that time.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes everything required into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` emits CSV `id,pulse,n_gates`, one row per instance. See `bench/io_spec.md` for JSON gate fields and CSV quoting.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`. Paths passed by the harness are absolute; output names may lack extensions.
- Use `bench status`, `bench eval-dev`, `bench validate-output FILE`, and `bench submit --final`.
- Dependencies are pinned in `code/env.lock`; PyPI `name==version` packages are supported. No editable, local or git installs.

## 6. Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Minimize hybrid infidelity of the specified sign-controlled qubit rotation under the fixed conditional-displacement amplitude budget.

**Fixed research objects:**

- Supplied coherent amplitude/angle; oscillator/qubit conventions, both coherent branches, even-cat input and prescribed targets.
- CD and unconditional qubit-rotation gate set, unit axes, total absolute CD amplitude <=pi/4 and Delta=1.
- 400/600 cutoff convergence test, worst-of-three-input infidelity and mean across instances.

**Editable methods:**

- Pulse sequence, gate count, allowed axes/angles/amplitudes within the fixed gate set and amplitude budget.
- Analytical designs, optimization, reusable learned design policies, search and implementation; two-gate GCR is a baseline.

**Forbidden substitutions:**

- Change target transformation, coherent/cat inputs, physical simulator, gate set or amplitude budget.
- Replace fixed cutoff/fidelity with an easier evaluation or submit fabricated states/scores instead of pulses.
- Use stored per-instance solutions, withheld-collection access or ID-dependent answers.

Pulse design and optimization under `code/` may change within the research scope above, while preserving the interface and physical guards. Data, frozen evaluator, README and submission format cannot be changed.

| Rule | Enforcement |
|---|---|
| Respect gate set, amplitude and numerical convergence | scorer |
| Use only supplied instance parameters at prediction time; do not branch on ids or memorize instance-specific solutions | patch audit |
| No reading, predicting or reconstructing withheld instance parameters outside the supplied input | patch audit |

Analytical pulse formulas and reusable learned policies are allowed; precomputed per-instance solutions and fabricated scores are disallowed. Train/predict run without network. Any randomness must honor BENCH_SEED. CPU budget is 8 and memory limit is 16 GB.

## 7. Known pitfalls
Gate lists are in execution order, so the momentum gate precedes position for GCR. Tensor order is oscillator first, and the qubit starts in its ground state. The quadratures are x=(a+a.dagger)/2 and p=-i(a-a.dagger)/2. The target rotation is rx(theta)=exp(-i theta sigma_x/2). Both coherent branches and the cat are scored, including relative phase.

Reducing oscillator cutoff inside your development code cannot alter the frozen 400/600 evaluation. BB1 at the public validation instance exceeds the CD budget. Negative lambda still costs abs(lambda); a nonunit axis cannot hide extra amplitude. Empty pulses are permitted but are scored as identity. The retained auxiliary plotting driver requires optional PDF dependencies not installed in the minimal runtime; train/predict and scoring need no PDF tools.

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
