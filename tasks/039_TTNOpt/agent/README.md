# TTNOpt: high-rank quantics tensor compression

## 1. Goal
Compress an observed 24-leg quantics tensor into a binary tree tensor network (TTN)
with auxiliary bond dimensions at most 4. Maximize `fidelity`, the squared overlap
of the normalized target and your reconstructed state, on unseen instances. The
research question is how tree structure and tensor updates improve compression at
a fixed bond budget. You may improve the algorithm under the fixed representation
and resource constraints below.

## 2. Data
The target is `f(x)=sum(j=1..30) cos(j*k_j dot x)`, with independent standard
normal components of each three-dimensional `k_j`. Each coordinate is the binary fraction represented by eight digits after the binary
point. Axes are x bits, y bits, z bits, most significant first.

- `data/train/README.md`: no training instances are supplied; the baseline saves a configuration.
- `data/val/input.npz` and `data/val/labels.npz`: two public validation instances, from seeds 12345 and 67890. Each contains `ids` and `target_i` arrays of shape `(2,)*24`.
- `bench/generator.py` defines the frozen distribution. The prediction input contains only observed target tensors and opaque IDs, with no generation seed or wavevectors. Labels contain the target for independent fidelity evaluation, not an optimal compressed state.

Do not alter or re-split the supplied data. There is no training split. Validation
is only for evaluation and method, hyperparameter or checkpoint selection. Do not
fit reusable models or learned preprocessing on validation tensors or labels.
Per-instance optimization on that instance's input tensor remains allowed.
At prediction time, each solution must depend only on that instance's observed
tensor, the validation-selected algorithm configuration and `BENCH_SEED`.

## 3. Metric and scoring
`fidelity` (higher is better) is the mean over instances of
`abs(vdot(Psi,phi))**2 / (vdot(Psi,Psi)*vdot(phi,phi))`, bounded by 1.
The frozen scorer contracts the submitted TTN into all 2^24 entries and restores
physical-axis order before computing the overlap. It ignores self-reported scores.
The score uses the final reconstructed state, not an average of local bond-update records.

The guard table and enforcement classes appear below.

Numerical or structural violations print `Infinity`, which the harness treats as
failure (the score-rule worst value is 0). Format errors fail validation. CPU and
memory limits remain binding task rules reviewed by patch audit.

The measured public baseline is **0.35592707073313595**, with population standard
deviation **0.00077052109041163** across three algorithm seeds. It is also reported
by `bench status`.
The judge runs three algorithm seeds, compares their mean against the archived
baseline, and requires relative improvement at least `max(delta,2*sigma/mean)`.
Delta is **0.005** (0.5% relative), rounded above twice the measured public
standard deviation divided by its mean (0.433%). The archived evaluation
variance can impose a larger noise threshold.

## 4. Baseline
Official code: https://github.com/Ryo-wtnb11/TTNOpt at commit
`0b18be209486a25f94d7e2220cfa7f01930279b2`. The implementation lives in
`code/ttnopt/src/FactorizeTensor.py`, `TwoSiteUpdater.py`, `DataEngine.py` and `TTN.py`.

The baseline starts from a matrix product network (MPN), a chain-shaped TTN,
constructed by sequential SVD with chi=4. It then runs fidelity-based two-site
structural optimization, type=1, beta=[0.04,0.02], chi=4 and up to 100 sweeps.
The supplied entry point does not forward the YAML convergence thresholds;
the actual defaults are both 1e-8. Only the chi=4 stage is used;
the chi=8/16 stages are outside this task.

The baseline solves each instance independently and emits its TTN tensors,
topology and center gauge. No fitted state is shared between instances.

The task allocation is 32 CPUs, allowing two independent 16-CPU solves. The memory
budget is 64 GB. Across the three reference runs, configuration writing took
0.56, 0.62 and 0.57 seconds. Validation prediction took 610.86, 617.84 and 631.54 seconds.
All tensor optimization happens during prediction. These are measured at 32 CPUs;
the memory budget is an allocation, not a measured peak.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes `config.json` into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes one NPZ TTN certificate per input ID, as specified in `bench/io_spec.md`.
- Honor `BENCH_SEED` and `BENCH_CPUS`; the baseline maps seed s to `(s-1) mod 2^32`, so seed 1 follows the authors' RNG seed 0. `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR` locate the supplied resources.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, `bench submit --final` provide the development workflow.
- Install dependencies using `uv pip install`; submission freezes `code/env.lock`. Only exact `name==version` PyPI pins can be rebuilt; editable, git and local installs are unsupported.

## 6. Research scope

The research objective and objects below are fixed; methods may change within this task scope.
The fixed tensor, axis order and bond budget make compression quality comparable across methods.

**Objective:** Maximize final-state dense fidelity compressing each supplied 24-leg quantics tensor to a binary TTN with all auxiliary dimensions <=4.

**Fixed research objects:**

- Supplied target tensor, 24 dimension-two physical axes, original bit order and instance distribution.
- Binary TTN representation with 22 degree-three nodes, complete physical axes and auxiliary bond dimension <=4.
- Final normalized dense-overlap fidelity against the original target, original instance/seed aggregation and isolation.

**Editable methods:**

- Tree topology/orientation, tensors, legal gauge, initialization, factorization/search/update rules, optimization surrogate and implementation.
- Any compression algorithm yielding the same admissible TTN certificate; sequential SVD, two-site updates and 100 sweeps are baseline choices, subject to the limit under Forbidden substitutions.

**Forbidden substitutions:**

- Change target/axes, raise bond dimensions, output nonbinary/cyclic networks or substitute a dense-copy representation.
- Replace final fidelity with averaged local-update records or a self-reported score.
- Memorize target solutions, reconstruct hidden seeds or use other evaluation instances.
- When retaining the supplied TTNOpt procedure, changing only fidelity.opt_structure.type=1, max_num_sweeps=[100], or setting beta=[0.04,0.02] to zero does not count as a method improvement.


| Guard | Requirement | Enforcement |
|---|---|---|
| Certificate archive | File size and expanded NPZ contents each at most 8 MiB; exact keys, IDs and tensor ranks | validator and scorer |
| Physical legs and tree | Exactly 24 dimension-two physical legs and 22 degree-three internal nodes; connected, acyclic | scorer |
| Bond budget | Every auxiliary dimension is at most 4; endpoint dimensions agree | scorer |
| Valid state | Correct center gauge; finite tensors and finite, positive reconstructed norm | scorer |
| Actual fidelity | Dense reconstruction against the supplied target; finite overlap in [0, 1+1e-12], clipped to 1 within that numerical tolerance | scorer |
| Resource budget | At most 16 CPUs per instance | patch audit |
| Instance isolation | No remembered solutions, hidden parameter access, seed inference, or use of other instances at prediction time | patch audit |


- Editable surface: TTN compression algorithms under `code/`, including hyperparameters within the limit under Forbidden substitutions, provided the fixed input/output interface and guards are honored.
- Frozen surface: `data/`, `bench/`, `README.md`, script signatures and prediction format.
- Total allocation: 32 CPUs and 64 GB memory budget. Each instance may use at most 16 CPUs; independent instances may run concurrently.
- Three algorithm seeds are evaluated; honor `BENCH_SEED` and do not select seeds using evaluation outputs.
- No network during training or prediction. Dependencies must be rebuildable from `env.lock`.
- Disqualifying: hard-coded or precomputed per-instance tensor values, paper-output lookup, guessing hidden generation seeds, reading unprovided parameters or hidden labels, using another instance to produce a solution, and bypassing the resource limit. Solve from the supplied target, not from a remembered instance identity.

## 7. Known pitfalls
The output path may have no extension: pass an open binary file handle to
`np.savez` so it does not append `.npz`. Object arrays and pickles are rejected.
Each node's third axis points toward the center; physical labels specify the
original target-axis order regardless of the optimized tree. A scalar fidelity
or a dense target copy is not a valid TTN certificate. A zero state is invalid.

All 100 sweeps can be expensive even if fidelity appears constant. Do not confuse
local bond-average records with the final contracted state. Limit numerical-library
threads per independent solve so concurrent solves stay within the CPU budget.
`PYTHONSAFEPATH=1` is enabled: entry scripts explicitly set `PYTHONPATH`.

Let an evaluation finish before editing its running shell entry points.

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
