# Bayesian mixture modeling with normalized inverse Gaussian weights

## 1. Goal
Infer the DMFM-IGau community posterior from each observed undirected network. Improve dARI, the mean posterior mean of the installed fossil adjusted Rand index, under the same fixed Bayesian model; higher is better. The baseline is the authors' telescoping sampler.

Given node allocations c and edge probabilities Q, each edge A_ij follows Bernoulli(Q[c_i,c_j]). Node allocations follow normalized inverse-Gaussian component weights. Each unnormalized weight has shape alpha/M and unit scale, then the weights are normalized to sum to one. M counts all components, k counts occupied components, and M_na counts empty components.

## 2. Data
There is no population training split. Each observed network is fitted independently; `data/train/README.txt` records this convention. `data/val/input.json` contains the full 50-network validation set; each graph has 50 nodes. `data/val/labels.json` supplies evaluation truth. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Fit no reusable model or learned preprocessing on validation, and carry no fitted state between independent networks. Per-network posterior inference from that network's own adjacency remains permitted.

`bench/generator.R` freezes the original graph generator, including its discarded diagonal RNG draws. Each observed graph is consistently permuted after generation to conceal the original fixed community ordering. Prediction inputs contain only opaque IDs and adjacency matrices; labels, parameters, seeds and permutations are separate. Do not re-split, relabel or import outside data.

## 3. Metric and scoring
`dARI` is higher-is-better. For each network, score all 50000 post-burn partitions using the installed `fossil::adj.rand.index`, then average those posterior means across networks. `bench/scorer.py` implements the literal fossil function. That function takes an absolute value and constructs its contingency table using the predicted cluster count on both axes; the one-cluster case consequently differs from common ARI implementations. These source behaviors are intentionally preserved. The metric implementation and aggregation are frozen.

Scorer/validator guards require exactly all supplied IDs, every post-burn iteration, 50 integer node labels per draw, canonical cluster labels 1..k, and consistent `1 <= k <= M <= 50`, `M_na=M-k`. Invalid output receives -1. The fixed DMFM-IGau posterior, including its likelihood and priors, is the research object. Inference algorithms and numerical approximations may change. The 50000 scored partitions must represent that posterior; their slot labels do not require the baseline MCMC transitions or burn-in. Repeating an arbitrary point estimate as a fabricated posterior is not permitted. All output slots and independent-network rules still apply. Posterior fidelity requires implementation audit beyond the CSV validator. The baseline's full settings are 100000 iterations, 50000 burn-in, BNB(1,4,3) on M-1, Beta(1,1) on Q and F(6,3) on shape.

Measured validation dARI is approximately 0.8617, with standard deviation 0.0005245222670093938 across reference repetitions. The configured relative improvement threshold remains 0.005.

## 4. Baseline
Official source: https://github.com/Fumiya-Iwashige/MFM-Inv-Ga at commit `d785705eb4c948eb227c567b760aeb7d7840726a`. `code/IG_functions_DMFM_network.R` defines `TS_IG_network_DMFM_BNB`; `code/IG_functions_general.R` implements its updates. The sampler updates node allocations, component counts, edge probabilities, unnormalized weights and the shape parameter within each network. These files are sourced from the editable candidate checkout on every prediction. `code/bench_predict.R` enforces the fixed F(6,3) shape prior.

Training records a protocol artifact without fitting. Prediction loads the supplied adjacency matrices, fits each network independently and writes the complete posterior CSV. BENCH_SEED controls chain randomness, and BENCH_CPUS limits independent workers. Protocol recording took 0.11 seconds in each reference repetition. Validation prediction took 1675.24, 1704.4 and 1745.33 seconds.

## 5. Interface (frozen)
Run `bash code/train.sh --out DIR`, then `bash code/predict.sh --model DIR --input PATH --output FILE`. The exact CSV contract is in `bench/io_spec.md`. `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR` and `BENCH_BENCH_DIR` are supplied by the harness.

`bench status`, `bench eval-dev`, `bench validate-output FILE` and `bench submit` use the frozen interface. The supplied read-only runtime provides R and its required installed libraries. System Python runs stdlib-only scorer/validator wrappers. `code/env.lock`, dependencies and the runtime are fixed.

## 6. Rules
The candidate-editable surface is `code/`; keep the script arguments, output contract and full evaluation protocol. Data, benchmark scripts, README, runtime snapshot and dependency lock are frozen. Return all full-input networks and every required partition output slot, using only that network's observed adjacency for fitting and prediction. Internet access is unavailable during fitting and prediction. Do not read hidden labels, embed predictions, infer truth from opaque IDs, or bypass input handling.

The task provides 32 CPUs and a 144 GB memory budget.

## 7. Known pitfalls
The metric averages ARIs over the complete submitted partition distribution; the output slots and averaging cannot be changed. Node order is the supplied adjacency order. Cluster labels must remain canonical, as required by the source fossil function. Independent workers each enable R JIT level 3 and use one numerical-library thread. Full posterior CSV output is large.

## Research contract

**Objective:** Improve posterior community inference for the specified DMFM-IGau network model, scored by posterior-mean fossil ARI.

**Fixed research objects:** The Bernoulli stochastic-block likelihood and normalized inverse-Gaussian DMFM weights remain fixed, with BNB(1,4,3) on M-1, Beta(1,1) on Q and F(6,3) on shape. Each observed 50-node network is independent; all 50000 posterior partition slots, canonical labels and k/M/M_na semantics, literal fossil ARI and aggregation remain fixed.

**Editable methods:** Any valid posterior sampler, deterministic integration or approximation strategy may replace telescoping Gibbs, with numerical optimization and initialization editable as part of that replacement. Iteration labels preserve the output grid, not a mandatory transition implementation or burn-in schedule.

**Forbidden substitutions:** Do not change weight/shape/component priors or graph likelihood, substitute generic spectral clustering for posterior inference, or fabricate a point-mass posterior by duplicating an unrelated point estimate. No hidden communities, generator-seed reconstruction, cross-network fitting or output-slot reduction. When retaining the supplied TS_IG_network_DMFM_BNB sampler, changing only s_pro=1.0, the initial k=10, iters=100000 or burn=50000 does not count as a method improvement.

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
