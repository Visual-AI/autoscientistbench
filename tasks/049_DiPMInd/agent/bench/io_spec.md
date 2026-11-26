# DiPMInd observed distance protocol

`train.sh --out DIR` prepares candidate artifacts. The baseline compiles the authors' bivariate Rcpp statistic into DIR/rcpp_cache and writes protocol.txt; no population training corpus is supplied.

`predict.sh --model DIR --input INPUT_DIR --output FILE` reads only the observations in INPUT_DIR. Each case's p_value may depend only on that case's Dx and Dy, artifacts from train.sh and randomization. The manifest.csv header is exactly `case_id,file,n`. Every n is 100; each file is case_id.rds. IDs are independently random opaque tokens, and rows are shuffled independently of scientific cell. Each RDS is an R list with exactly Dx and Dy, finite symmetric nonnegative 100 by 100 distance matrices with zero diagonal. Dx is the observed L1 distance of 200-node undirected preferential-attachment graph Laplacians; Dy is the observed L1 distance of Gaussian vectors. RDS inputs carry no model, rho, latent variables, generation RNG state or seed. The original statistic takes exactly these distance matrices as input.

Prediction CSV has exactly `case_id,p_value`, one row per observed case. p_value must be finite in [0,1]. Missing, extra or duplicate cases fail validation. The baseline prediction uses 500 permutations and (1 + exceedances)/(500 + 1), extracting the prod-AD statistic. Alternative valid observed-distance tests and calibration procedures are allowed. BENCH_SEED controls candidate randomness, BENCH_CPUS controls workers.

Separate labels map IDs to model/rho only. Full evaluation has 200 replications in each of 12 model/rho cells. Mean power averages the ten alternative-cell rates at p_value <= 0.05, subject to the Model 1 null rate <=0.10 and Model 2 null rate <=0.081.

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
