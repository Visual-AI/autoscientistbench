# Tensor estimation with ScaledGD

Research question: how few updates does it take to recover a low-rank tensor from corrupted, incomplete or compressed observations, across condition numbers kappa? The paper (Wu, JMLR 2026) proposes ScaledGD: gradient descent on the t-SVD factors L and R from a spectral initialization, each gradient scaled by the inverse Gram matrix of the other factor. It proves a linear rate independent of kappa at the per-iteration cost of gradient descent. Its Figure 1 compares ScaledGD with vanilla GD from the same initialization, by relative error against iteration count.

Minimize `scaledgd_iterations`: the sum of first update indices reaching relative Frobenius error at most 1e-6 in the 32 Figure 1 cells.

## Experiment

Observations are noiseless. Each of four families runs with DFT and DCT transforms at condition numbers 1/5/10/20.

- Tensor RPCA: `Y = X* + S*`, 10% of entries corrupted.
- Tensor completion: `Y = Omega.*X*`, entries observed with probability 0.4.
- Robust tensor completion: `Y = Omega.*(X* + S*)`, 10% corrupted, observed with probability 0.6.
- Tensor regression: m=25000 inner products of `X*` with Gaussian tensors.

Regression uses n=50, n3=20, r=5; the others use n=n3=100, r=10. `X*` has tubal rank r, singular values running linearly from 1 to 1/kappa in every slice of the transform domain, and singular tensors taken from random-sign tensors.

## Baseline

The baseline solver is `code/tensor_policy.py`, using update formulas in `code/protocol30.py`. Baseline eta is 0.5 (0.4 for robust completion), z0=z1=0.5 and decay=0.95; RPCA and robust completion soft-threshold the residual at z1*decay^t. `ScaledGD` names the primary scored trajectory in the protocol; it does not require that algorithm. Vanilla GD is a secondary diagnostic.

Baseline validation score: 7568, standard deviation 0 (one algorithm seed; the updates use no random numbers). On 128 CPUs: `train.sh` 10.83 s, `predict.sh` 0.72 s (packaging only), full validation replay in the scorer 3365 s, about 56 minutes, running ScaledGD and GD for every cell.

## Score and guards

The initial estimate is update zero; at most 1000 error records cover updates 0 through 999. If any cell fails to reach the target, the score is 32000. A protocol violation, failed worker or incomplete solver scores Infinity. The outer improvement criterion is 1%. Preparation, prediction and infrastructure times are recorded as diagnostics and do not change the score, provided replay finishes inside the overall evaluation deadline.

- `validate.py`: artifact format.
- Scorer: allowed oracle method and case, consecutive `t`, at most 1000 records, matching shape, finite values, a ScaledGD record for every cell, sealed primary trajectories, completion receipt. The score comes from oracle records only; forged curves, counters and direct prediction NPZ files are rejected.
- Patch audit: every actual update reported, legal inputs and oracle feedback, no fitting on validation.

## Error oracle

The error oracle lives outside candidate processes. A worker's input holds observations but no ground-truth tensor, generation seed, generator RNG state or truth-dependent source. Regression also receives spectral initial factors; in small cells the factor arrays are zero and the baseline computes its spectral initialization from `Y`. Call the frozen `bench_observe(X,t,method,case_index)` after every iterate. It returns the independently computed relative error and records the call; `t` starts at 1. Hiding work between oracle calls is prohibited.

The regression baseline advances ScaledGD and GD in the original shared batch at each step. Legal GD feedback is available in that order; observing GD before the primary trajectory ends is permitted, within the oracle rule under Forbidden substitutions. Every actual update must be reported; candidate methods must respect the input and oracle-access rules.

## Interface

`bash code/train.sh --out DIR` prepares configuration and the offline Octave 9.4 runtime. `bash code/predict.sh --model DIR --input PATH --output FILE` packages the submitted code and pinned Python dependencies into a `private-policy-v2` artifact. Its input is a descriptor with no evaluation seeds or observations. The task scorer first replays the same submitted `train.sh` in a timed isolated worker using the original data and algorithm seed, then calls `policy.sh` using only that newly prepared model. Outer model files and `train_seconds.json` are not used. Put every reproducible preparation step in `train.sh`; its outputs remain available to the solver. See `bench/io_spec.md` for the interface.

## Data

Training split: `data/train/input.npz` (32 cell descriptors with their seed, no truth) and `data/train/runtime.tar.gz`. Validation split: `data/val/request.json` starts validation; `data/val/evaluation/{input.npz,labels.npz,replay.json,runtime.tar.gz}` holds its descriptors, truth, settings and runtime. `bench/generator.py` and `bench/tensor_materialize.py` hold the frozen instance generation.

Validation truth is for evaluation and for method, hyperparameter or checkpoint selection only. Solving a cell from its own observations is the per-instance fitting this task defines. Hidden evaluation observations are generated from independently held seeds in the same frozen family. Do not memorize validation truth, infer private seeds, reconstruct hidden truth from generator state or access oracle internals.

## Resources and isolation

Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_CODE_DIR` and `BENCH_BENCH_DIR`. Dependencies remain exact pins in `code/env.lock`. CPU budget is 128, memory budget 64 GB. Regression uses up to 24 assigned CPU cores; small cells use at most two each. Full regression retains the dense 50000-by-25000 Gaussian operator. No network or runtime downloads are needed.

`data/`, `bench/`, this README, the observation whitelist and evaluation protocol are frozen. Candidate execution has fresh temporary storage and process/IPC/network namespaces, and read-only code, model, observations and `data/`. Write scratch files under the supplied temporary directory or output directory. The bundled oneMKL loader uses static compatibility files that expose no real process maps, environment, file descriptors or host process tree. System BLAS/LAPACK alternatives are mounted read-only.

## Research scope

**Objective.** Recover the fixed low-rank tensors in all 32 inverse-problem cells with as few fully reported updates as possible, reaching the original relative Frobenius error threshold.

**Fixed research objects.** Four inverse-problem families, DFT/DCT transforms, dimensions/ranks/condition numbers, original observations and supplied initial-factor inputs; paired regression operator/observations across its cells, private target tensors and permitted error-oracle feedback; relative Frobenius error at most 1e-6, all 32 cells, complete initial/update reporting, at most 1000 records and original first-reaching-index aggregation.

**Editable methods.** Numerical tensor-recovery solver, lawful initialization computed from allowed inputs, preconditioning, training/optimization surrogates and hyperparameters such as the step size; implementation and optimizer while honestly reporting all internal solver updates.

**Forbidden substitutions.** In the regression batch, a cell's iterates may use only the shared operator and that cell's own `y`, initial factors and oracle errors; nothing derived from another cell may enter them. Oracle errors may be used only to decide when to stop a trajectory, and the `GD` slot may carry only a vanilla GD trajectory; neither may choose a step, hyperparameter, direction or iterate. An update may apply the forward operator and its adjoint at most once, and the initialization may apply them at most once plus one t-SVD; anything more counts as further updates and must be reported. Substituting an easier target tensor, forward operator, transform, rank/dimension or instance collection; omitting internal updates, falsifying convergence curves, recovering hidden target/generator state or using unpermitted oracle access; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. The scorer generates the observations, computes the curves and scores against its own truth; the rest relies on patch audit.

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
