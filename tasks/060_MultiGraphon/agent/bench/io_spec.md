# Graphon estimation I/O

Each instance is a collection of 200 undirected binary networks with sizes uniformly drawn from integers 10 through 100. IDs are opaque and row order has no model meaning. NPZ files never require pickle.

Input keys:

| Key | Shape/type | Meaning |
|---|---|---|
| id | C strings | Unique collection identifiers |
| vec_n | C by 200 integers | Observed graph sizes, original graph order |
| offsets | C+1 integers | Offsets into adjacency, starting at zero |
| adjacency | flat uint8 | Row-major full square adjacency matrices concatenated by collection, then graph; offsets delimit collections |

The true graphon, latent node positions, model identity and generation seeds are not inputs. `format.unpack_collection` returns the observed matrices as integer arrays without altering them.

Prediction keys (NPZ, written to the exact output path through a binary file handle):

| Key | Shape/type | Meaning |
|---|---|---|
| id | C strings | Exactly the input ID set, any order, no duplicates |
| k | C integers | Number of equal-width blocks, between 1 and 1000 |
| offsets | C+1 integers | Starts at 0; successive differences are k squared |
| values | flat floats | Concatenated row-major k by k symmetric graphon probability matrices, finite and in [0,1] |
| time_s | C floats | Positive finite wall seconds spent estimating each collection, including all method computation; excludes input decoding, imports, process startup and output serialization |

No prediction-derived preprocessing may be hidden outside the measured interval. Workers are single-threaded and their count is bounded by BENCH_CPUS. The matrix diagonal is a within-block edge probability, not forcibly zero. Values specify the step function; do not bilinearly resize the baseline estimate.

Public validation labels `truth.npz` contain `id,graphon_id,seed,repetition,reference_id,reference`: zero-based graphon identity, generation seed, trial index, and the 1000 by 1000 reference matrices (zero diagonals before degree sorting). They also carry baseline per-collection timing and errors for diagnosis, with no baseline-relative scoring thresholds. Labels may be used for offline validation only.

`validate.py PRED INPUT` checks keys, exact IDs, finite values, shapes, symmetry and probability bounds. `scorer.py PRED LABELS` computes the unchanged degree-sorted step-function off-diagonal L2 error, times 1000, averaging trials then graphons. Elapsed estimation time is diagnostic. Invalid files return Infinity. Reported time alone is not trusted. Timing scope, collection independence and truth-table avoidance require patch audit.

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
