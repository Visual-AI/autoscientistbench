# BFM-SWM: budget-feasible procurement for submodular welfare

Protocol version 2 evaluates a submitted policy against a private supplier service. The outer commands remain `train.sh`, `predict.sh`, `validate.py` and `scorer.py`.

## 1. Goal

A buyer with a budget wants to hire suppliers in a social network to spread information. The buyer values a set S at `v(S) = |N(S)|`, its covered neighbours. Each supplier has a private cost, and the social welfare is `v(S) - c(S)`. The paper's BFM-SWM (ICML 2026) is a budget-feasible descending clock auction for this problem, with worst-case guarantees and the economic properties below.

Improve the welfare that such a mechanism achieves. Your policy never sees a cost. It can only make take-it-or-leave-it offers and read accept or reject. Use the graph and budget in each instance input and obtain cost information through the offer service. Do not rely on any particular cost model.

## 2. Metric and guards

`welfare_ratio`, higher is better: the mean over instances of `(v(S) - c(S)) / G`, where S is the declared winner set and G is a frozen full-information benchmark (lazy greedy on the true costs with `c(S) <= budget`, the better of largest `mg/c` and largest `mg-c`). The ratio can exceed 1, because greedy is not optimal. The judge computes it on a held-out split and compares it with the archived baseline; PASS needs a relative improvement of at least `delta = 0.25`.

| Guard | If broken | Enforced by |
|---|---|---|
| A supplier answers only accept or reject, by `cost <= price` | not breakable from a worker | scorer: costs stay in its process |
| Prices to a supplier never rise | `Infinity` | scorer |
| A rejection is final: no further offer, and no win | `Infinity` | scorer |
| Each winner accepted its most recent offer, and that offer is its payment | `Infinity` | scorer |
| Total payment at most the budget, tolerance `1e-9` | `Infinity` | scorer |
| Non-negative surplus, `v(S) - payment >= 0`, tolerance `1e-9` | `Infinity` | scorer |
| One declaration per instance, no calls after it; every worker exits cleanly | `Infinity` | scorer |
| Fresh worker per instance, no shared files, no network | a policy cannot carry state across | scorer's isolation |
| No learning of costs except through offers; no memorized validation prices; fitting only on train | disqualifying | patch audit |

Truthfulness and individual rationality need no separate check, because they follow from the protocol. A supplier sees a non-rising sequence of prices, and rejecting ends its participation. So accepting exactly while the price covers its cost is its best reply whatever the mechanism does. A winner is paid an offer it accepted, so at least its cost. A candidate does not have to keep the paper's approximation theorem, but it must keep this protocol.

## Research contract

### Objective

Raise the social welfare `v(S) - c(S)` of a budget-feasible procurement mechanism on coverage valuations with private supplier costs. Use only take-it-or-leave-it offers with non-rising prices. The fixed objects below keep comparisons under the same information and payment constraints.

### Fixed research objects

- The procurement setting: coverage valuation on the supplied graph, private supplier costs, one budget per instance, welfare as the objective.
- The offer protocol and the economic guards above, including the payment rule.
- The metric, its frozen benchmark, the trusted oracle and scorer, and the public data.

### Editable methods

- Everything inside the mechanism: the order of offers, the price rule and its parameters (alpha, beta, epsilon, or none of them), rounds, candidate sets, the stopping rule, and the choice of final winners.
- Learning within an instance from accept and reject answers, including deliberately sacrificing suppliers to learn about the cost distribution. Randomization through `BENCH_SEED` and `algorithm_index`.
- Anything fitted offline on the training cost draws and frozen into the model directory. The implementation language and data structures.

### Forbidden substitutions

Learning costs other than through offers (files, memory, processes, timing, or regenerating draws). Changing the oracle, scorer, value function, budgets, benchmark or supplier behaviour. Simulating another payment rule such as pay-as-bid. Carrying state between instances. Memorizing the public validation draws as a price table. Training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

## 5. Baseline

BFM-SWM by the authors, official code <https://github.com/xue74193-dot/BFM-SWM> at commit `82f97be91f4404a9f943c5ebf3d60a272a76a13d`, under `code/Influence Maximization/`, with `alpha = 1 + sqrt(6)/2`, `beta = 3`, `eps = 0.1`, `l = 1`.

The baseline uses `bench_policy.cpp` as its worker entry point. `BFM-SWM.h` obtains cost information through the offer service and batches offers with `stop_on_accept`. `main.cpp` and the `Deng-*.h` headers implement sealed-bid comparisons that require explicit bids and are not compiled. `Crowdsourcing/` is a separate application and is not used.

## 6. Interface

- `bash code/train.sh --out DIR` compiles the mechanism into DIR. Keep anything your policy needs in DIR; `g++` is available and `$BENCH_BENCH_DIR` is readable here.
- `bash code/predict.sh --model DIR --input REQUEST --output FILE` seals `code/` and DIR into the policy artifact. It runs no instance.
- `code/policy.sh --model DIR --input INSTANCE --output FILE` is the per-instance entry point, started by the scorer in a fresh isolated worker. It is yours to replace. The baseline execs a compiled binary; a Python policy may `from offer_client import read_instance, load_graph, offer, declare`.
- `bench/io_spec.md` has the input format, the service calls and the rules.
- Environment in the worker: `BENCH_RPC_SOCKET`, `BENCH_DATA_DIR` (graphs only), `BENCH_MODEL_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`, `BENCH_SEED`, `BENCH_CPUS`.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, `bench submit [--final] [-m MSG]` work as usual. `python bench/scorer.py ARTIFACT data/train/labels` scores an artifact on the training draws.

**Data.** `data/graphs/{slashdot,epinions}` are the public graphs in the authors' format. `data/train/labels` has 80 instances and `data/val/labels` has 60: each is a graph, a cost draw and a budget from the paper's grid 100..1000, with two public cost families. The labels are public and include the costs, so you can study them and simulate offline. Only the training draws may be fitted. The held-out split is held by the judge and is never readable from this workspace.

## 7. Resources

4 CPUs and 8 GB of memory. One worker runs per instance on one CPU, up to 4 at a time, so `BENCH_CPUS` is 1 inside a worker. There is no network during train, predict or policy runs.

Measured baseline validation `welfare_ratio`: mean 0.5648436056480995 and standard deviation 0.0 for `BENCH_SEED=1`. Compilation took 3.07 s and validation artifact sealing took 1.22 s. The validation worker phase took 22.78 s across 60 instances, with mean worker time 1.48 s, maximum 2.25 s, about 12.6 million offers and 4,900 RPC calls. Every worker start, graph load and RPC call counts against your overall time budget, so prefer batches with `stop_on_accept`. There is no separate per-instance time limit. RPC calls remain subject to the service safety limit in `bench/private_eval.py`.

## 8. Seeds

The judge uses one seed: the baseline is deterministic, and parity is checked at a relative tolerance of `1e-9`. If your policy is random, derive its randomness from `BENCH_SEED` and `algorithm_index` so a rerun repeats exactly. Costs are stored as fixed draws, so nothing about them depends on `BENCH_SEED`.

## 9. Notes

- `S_star` selection in the baseline compares `v - p`, never `v - c`, because that is all a mechanism can know. `welfare_ratio` is computed from the true costs only by the scorer, after your declaration.
- A price of 0 is a legal offer and is rejected by any supplier with a positive cost. That is how the baseline drops zero-marginal suppliers.
- Prices must be finite and non-negative. The authors store them as 32-bit `float`, which rounds; a price you recompute must not round above the previous offer.
- A supplier who accepted is not committed: you may offer it less later, or not declare it. Choosing a winner after lowering its price means paying the lower price, never an earlier one.

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
