# 063_MartingalePost interface
Input directory has index.csv columns id,n,offset,file. Each row points to a 2xn Float64 column-major matrix in a binary shard; offset is bytes. Host is little-endian x86_64. n is one of 20,100,500. Only observations and layout metadata appear; truth and generator seeds are separate.
Output JSON: elapsed_s plus predictions list, each {id,lo:[mu1,s1,s12],hi:[mu1,s1,s12]}. Exactly one finite ordered interval triple per input ID. Timing covers predictor Julia startup, loading and computation. Elapsed time is diagnostic; its provenance and absence of truth access require patch audit.
Validation contains 5000 datasets per n; tiny independent smoke contains one per n; the baseline smoke uses B=2000 / T=50. Other valid numerical methods may compute the same central 95% intervals of the fixed normal parametric martingale posterior. Generic confidence intervals or another posterior are not equivalent outputs.

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
