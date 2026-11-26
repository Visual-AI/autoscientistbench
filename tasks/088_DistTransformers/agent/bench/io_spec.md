# I/O specification

Input is an NPZ archive with `id` (n strings), `phi` (n,2 float32: supplied prior concentration and rate), and `z` (n,1 float32: the observed normal response). IDs and row order are opaque. The unknown variance, posterior parameters, sampling states and evaluation seeds are absent. Supplied prior parameters are observable conditioning information required by prior adaptation, not the hidden variance.

Output is CSV, exactly one row per input ID, in any order. Header:

```
id,w0,mu0,sd0,w1,mu1,sd1,n_components,amortised,forward_s,baseline_s
```

Each row describes two Gaussians in log-variance space. Weights are strictly positive and sum to one within 2e-6; standard deviations are positive. All numbers are finite. `n_components=2`; `amortised=1` is required. Optional `model_api.AMORTISED` must be 1 (the baseline default); this declaration alone does not establish algorithmic compliance. Times are positive seconds per 1000 problems, measured by one frozen-runner forward call; the second time is an unmodified DT architecture in the same process. Model loading, GMM decoding, scoring, logging and I/O are outside that diagnostic region.

`train.sh --out DIR` must save all fitted state. The baseline writes `model.pt`, `config.json`, and `certificate.json`. Candidate code exposes `model_api.load_model(model_dir)` returning a torch module or a serializable fitted-state object, and `model_api.forward(model, phi, z)` returning a NumPy array or Torch tensor `(n,2,3)`, with final axis `(weight, mean, standard deviation)`. Any shared amortised model and training surrogate may implement this interface. Forward applies the learned mapping using only each row's conditioning information and fitted state. It must not fit per-instance parameters, run a posterior-sampling chain, mutate fitted state or learn across independent evaluation rows. The frozen runner checks the representation and state immutability and writes the CSV. Keep `predict.sh` invoking this runner. Predictions must agree under replay and row permutation; randomized inference must use reproducible randomness derived only from permitted inputs and the algorithm seed.

Labels (`truth.npz`, public for validation) contain `id,a,b,x,draw,position,posterior_rng,seeds,n_mc,baseline_forward_s`. Here `a,b` parameterize the exact inverse-gamma posterior, `x` is the latent variance, and the remaining fields restore the original Monte Carlo sample stream and aggregation order. The scorer restores each draw's original order before sampling. `baseline_forward_s` archives a conversion-time measurement for diagnostics.

`python bench/validate.py PRED INPUT` checks the format, IDs and finite values. `python bench/scorer.py PRED LABELS` prints one JSON object with key `posterior_kl` (lower is better; `Infinity` on violation). Monte Carlo KL uses the original float32 implementation with the log-transform Jacobian and 10000 samples per problem. Finite negative estimates are not clamped.

`python bench/check_model.py MODEL --input INPUT --pred PRED --code CODE` independently replays the fitted model through the frozen runner and a permuted subset, and checks per-problem prediction consistency. The operator must run this probe in the execution sandbox. Raw CSV claims alone cannot prove input provenance or independent inference; these also require patch audit.

## Research scope

Improve prior-flexible amortised two-component approximation to the specified inverse-gamma posterior. The supplied prior/observation distribution, normal likelihood, latent variance and log-variance output space are fixed. Inference is amortised across problems; output is exactly two valid Gaussians. The original posterior KL Monte Carlo formula, Jacobian and aggregation remain fixed.

The shared inference architecture, fitted representation, offline training method, training surrogate and optimizer may change; a Transformer is not mandatory.

Do not replace the Bayesian problem, fit each evaluation problem separately, run a posterior-sampling chain, expand the mixture family, change KL scoring, access labels or transfer information across evaluation problems.

The operator replay checks output consistency and fitted-state immutability. The declaration does not prove amortisation; actual source execution and data provenance require audit.

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
