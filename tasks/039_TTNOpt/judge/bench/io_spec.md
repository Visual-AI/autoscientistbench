# Frozen interface: quantics TTN compression

Input is a NumPy NPZ archive, loaded with `allow_pickle=False`. `ids` is a unique
one-dimensional string array. `target_i` is the observed float64 array of shape
`(2,)*24` for `ids[i]`. Axes are x bits (most significant first), y bits, z bits,
eight each. Targets contain unnormalized values of the cosine-sum function defined in `README.md`. No generating seed or
wavevector is supplied. Labels have the same keys; fidelity has no precomputed
optimal answer and is evaluated against the target itself.

Output is NPZ written to the exact supplied path, with these keys only:

- `ids`: exactly the input ID set, without duplicates; order is arbitrary.
- `edges_i`: integer array `(22,3)` for output `ids[i]`.
- `gauge_i`: finite floating or complex rank-two center tensor.
- `tensor_i_j` for j=0..21: finite floating or complex rank-three node tensor.

Edge labels 0..23 designate physical axes, each appears once and has dimension 2.
Labels 24..44 designate auxiliary bonds, each appears at two distinct nodes with
equal dimensions, at most 4. Every internal node has three distinct incident edges.
The whole network must be a connected binary tree. Tensor axes follow each row of
`edges_i`. Axis 2 points toward the center. Exactly one auxiliary edge appears as
axis 2 on both its endpoints; `gauge_i` connects these axes, in increasing node
index order. Other edges connect a child's axis 2 to a parent's axis 0 or 1.
Every binary tree can be oriented this way. Canonical/isometric gauge is optional.
The reconstructed state must have finite positive norm. Global scale cancels.

The validator checks format, IDs, finite values and ranks; the scorer checks all
physical/tree/bond constraints and contracts all 2^24 entries. Archives and their
uncompressed contents are limited to 8 MiB. No pickle/object arrays or extra keys.
Write NPZ through an open binary file handle to avoid adding a filename suffix.

`train.sh --out DIR` saves any artifacts required by prediction; the baseline saves `config.json`.
`predict.sh --model DIR --input PATH --output FILE` solves each observed instance.
The baseline maps `BENCH_SEED=s` to the authors' optimizer RNG seed `(s-1) mod 2^32`; alternative methods must honor `BENCH_SEED`.
`BENCH_CPUS` limits the total allocation. Each instance may use at most 16 CPUs; independent instances may run concurrently within the total budget.

## Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Maximize final-state dense fidelity compressing each supplied 24-leg quantics tensor to a binary TTN with all auxiliary dimensions <=4.

**Fixed research objects and controls:**

- Supplied target tensor, 24 dimension-two physical axes, original bit order and instance distribution.
- Binary TTN representation with 22 degree-three nodes, complete physical axes and auxiliary bond dimension <=4.
- Final normalized dense-overlap fidelity against the original target, original instance/seed aggregation and isolation.

**Editable components:**

- Tree topology/orientation, tensors, legal gauge, initialization, factorization/search/update rules, optimization surrogate and implementation.
- Any compression algorithm yielding the same admissible TTN certificate; sequential SVD, two-site updates and 100 sweeps are baseline choices, subject to the limit under Forbidden substitutions.

**Forbidden substitutions:**

- Change target/axes, raise bond dimensions, output nonbinary/cyclic networks or substitute a dense-copy representation.
- Replace final fidelity with averaged local-update records or a self-reported score.
- Memorize target solutions, reconstruct hidden seeds or use other evaluation instances.
- When retaining the supplied TTNOpt procedure, changing only fidelity.opt_structure.type=1, max_num_sweeps=[100], or setting beta=[0.04,0.02] to zero does not count as a method improvement.

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
