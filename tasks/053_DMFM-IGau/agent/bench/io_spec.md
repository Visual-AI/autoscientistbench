# 053_DMFM-IGau network posterior interface

`train.sh --out DIR` records the seed and protocol. `predict.sh --model DIR --input PATH --output FILE` performs fresh independent community inference for each supplied network. The DMFM-IGau likelihood and priors are fixed; posterior inference and approximation methods are editable.

Input is JSON: `{"schema":"sta26-network-v1","networks":[{"id":"32 lowercase hex characters","adjacency":[[0,1,...],...]}]}`. Each graph has 50 nodes, an undirected binary adjacency matrix, and a zero diagonal. No true community, graph-generation seed, parameter, or original node index is an input. Row order defines node order.

Output is UTF-8 CSV with the exact header `id,iteration,k,M,M_na,c1,...,c50` (expanded through c50). Emit networks in input order. For each network emit output slots numbered 50001 through 100000 in order. These identifiers retain the original evaluation grid; they do not require a particular MCMC transition or burn-in schedule. Each row contains 50 integer cluster assignments, canonically numbered 1 through k; `1 <= k <= M <= 50` and `M_na = M-k`. The validator rejects missing, extra, duplicate, reordered, noninteger or nonfinite values and inconsistent cluster/component counts. All slots must describe the specified DMFM-IGau posterior. Numerical inference need not be the original Gibbs sampler, but an arbitrary deterministic community estimate cannot be relabeled as posterior draws.

The frozen scorer computes the installed `fossil::adj.rand.index` for each retained partition against the corresponding true communities, averages all 50000 draws per network, then averages networks. It preserves fossil's absolute-value and contingency-table semantics. Invalid output scores -1.

Ordinary inputs require the full schema and all 50000 scored partition slots.

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
