# Input and output

`--input` is a NumPy archive with exactly `id` (n unique opaque strings) and
`x` (n, 200, 20) float64 observations in chronological order. No latent factors,
loadings, simulation seeds, or other latent parameters are inputs. Each matrix
is an independent instance. Do not combine observations across instances.

Output is UTF-8 CSV, including when the output filename has no extension.
Header: `id,t,f1,f2,f3,f4,f5`. Each supplied id requires exactly 200 rows, with
`t` ranging from 0 through 199. Factor entries must be finite numbers. Rows can
appear in any order; duplicate or missing (id,t) pairs and unknown ids fail.
No alignment, padding, truncation, or dropping failed simulations is allowed.

`truth.npz` contains `id`, `factors` (n,200,5), `seed`, `loadings`, `shocks`,
`noise_draws`, and `noise`. These are scoring labels, not prediction features.
Validation labels are public. Labels for the evaluation instances are held by
the judge. The scorer averages one MCC per supplied instance.

`train.sh --out DIR` saves configuration needed for per-instance fitting;
`predict.sh --model DIR --input PATH --output FILE` fits independently on the
observations and emits recovered factors. BENCH_SEED controls algorithmic
randomness, separately from the simulation RNG. BENCH_CPUS bounds the total CPU allocation across workers and numerical threads.

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
