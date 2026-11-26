# BMVL private evaluation I/O

The four external calls retain the usual train, predict, validate and score signatures. Outer input is a JSON request with `task_id="cvpr-427-BMVL"`. Outer prediction is a stored ZIP in canonical `private-policy-v2` format, containing candidate `code/`, `model/` and changed locked dependencies. An exact match with the frozen reference lock reuses the trusted reference runtime packages; modified locks export their packages. The validator checks structure without running candidate code. Real datasets and labels are never inputs to outer prediction.

A private worker runs `bash /ws/code/policy.sh --model /ws/model --input /ws/input --output /ws/output/pred.out` in a fresh filesystem and network sandbox. Candidate source and prepared model files are read-only. Only that request's input is mounted; writable temporary/output locations are unique to this invocation. No worker has other folds' data or state. Worker logs and fit artifacts never become public outputs.

For `stage="fit"`, `/ws/input/request.json` contains `num_views`, `num_classes` and `fit_seed`. `/ws/input/train.npz` contains float32 arrays `x0`, `x1`, …, and integer target vector `y`, in the original training-fold order. The worker writes one opaque fit artifact. The scorer copies it without importing candidate code or unpickling model contents.

For `stage="predict"`, the request contains `num_views` and `num_classes`. `/ws/input/fit.out` contains that fold's private fit artifact. `/ws/input/input.npz` contains only the observed feature arrays, in scoring order. There are no labels, original row IDs, noise indicators or clean alignment. The clean and noisy conditions execute in separate fresh workers, each receiving the same fit artifact. Prediction output must be JSON `{"pred":[0,1,...]}`, exactly one integer class index per row, in `[0,num_classes)`, with no extra keys.

The scorer always evaluates all five original datasets, all ten original seed partitions and both 0% and 50% conditions for formal evaluation. Each seed accuracy percent is rounded to two decimals before averaging seeds and then datasets equally. `acc_eta50` is the noisy mean if `acc_eta0>=91.80`, else zero. Invalid outputs or private execution failures fail scoring and reveal only a fixed safe error reason. The public development manifest uses the same private interfaces on the real CCV dataset with seeds 0 to 4 and a clean floor of 58.00; it is never a benchmark reference.

The original scientific inputs and metric are frozen. Candidate model/representation/loss/optimizer/training schedule may change within the same noisy-correspondence classification objective. Cross-fold memory, access to both test conditions in one inference process, external data and hidden-label access are prohibited.

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
