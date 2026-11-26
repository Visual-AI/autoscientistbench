# Input and certificate format

Input is NPZ (allow_pickle=False): `id` int64 [N], strictly increasing; `x` float32
[N,3,32,32], CIFAR-10 pixels normalized channelwise with mean
(0.4914,0.4822,0.4465) and standard deviation (0.2470,0.2435,0.2616).
Training `train/train.npz` has `id`, `x` [N,3,40,40], `y` int64 [N]. The normalized
padding corresponds to raw zero padding. Apply the original random crop and flip.
Validation labels contain `id`, `y`, `x` and `dense_hash_S` Unicode scalar entries
for the approved dense checkpoint for each seed S. Dense hashes include tensor
names, shapes, dtypes and contiguous bytes, in sorted parameter-name order.
The approved dense checkpoint itself is a read-only input: `dense/seed<S>.npz` holds
the 24 FC12 tensors under their parameter names (`net.J.weight`, `net.J.bias`,
float32) plus `selected_epoch` int64, and its digest is `dense_hash_S`. Load the
file for BENCH_SEED; it is the network to prune, not a target to reproduce.

Prediction is NPZ written through an open binary file handle, with these exact keys:

- `id` int64 [N], exactly the input ids in the input order, no duplicates.
- `logits` float32 [N,10], finite unnormalized class scores.
- `seed` int64 scalar, equal to BENCH_SEED.
- `dense::net.J.weight`, `dense::net.J.bias`, `pruned::net.J.weight`,
  `pruned::net.J.bias` float32 arrays, for J=0,2,...,22.
- `mask::net.J.weight` boolean matrices, for J=0,2,...,22.

FC12 dimensions are (3072,1000,900,800,750,700,650,600,500,400,200,100,10).
Each weight matrix has shape [output_width,input_width], bias [output_width].
There is an in-place ReLU after every Linear except the last. The input is flattened
in channel-first order. `mask=True` means retained. Biases are outside the sparsity
denominator. There are 7,263,000 Linear weights.

The dense state must match the frozen checkpoint for the submitted seed. Every
retained weight and every bias must equal dense exactly; removed weights must be zero.
The scorer recomputes both models on the supplied rows, verifies the submitted logits
(rtol 1e-5, atol 1e-4 for CPU reduction variation), and scores its own recomputation.
CE uses the author's float32 batch means with batch size 128 and sample-weighted
Python accumulation. Sparsity is 100*(1-kept/total). Epsilon is abs(pruned CE-dense CE).
Epsilon must be at most 0.1. Mathematical failure score is 0 (`raw_score`); the
`sparsity` key is Infinity on violations, the harness's explicit failed-run signal.

Model output must include `certificate.npz` (all prediction keys except `id,logits`)
and `model_card.json`: `format`="fc12-mask/v1", `certificate`="certificate.npz",
`seed`, `dense_sha256`. Additional card metadata are allowed. The optional
`check_model.py MODEL_DIR --input INPUT --pred PRED` checks model/prediction agreement;
all numerical feasibility constraints are already enforced by scorer.py.

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
