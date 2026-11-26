# Asymmetrically perturbed DGDA on Leduc

**Research question:** how low can the last-iterate NashConv of a two-player zero-sum extensive-form game go for a fixed number of counted strategy updates? The paper proposes asymmetric perturbation. For a small enough strength, perturbing only one player's payoff leaves that player's equilibrium strategy equal to an equilibrium strategy of the original game, while perturbing both players generally shifts it.

## Experiment

The paper's Figure 4 runs AsymP-DGDA against SymP-DGDA, DMWU, DGDA, DOMWU and DOGDA on five games, including Leduc poker. It plots the original-game NashConv of the last-iterate strategy pair against the total number of strategy updates. AsymP-DGDA uses one run per player, so its cost is summed over both runs and each run gets half of the budget. The authors tune eta and mu separately for each algorithm and each game (Appendix A.4, Table 4). CFR-family methods are compared separately on average iterates (Appendix A.5).

This task keeps that measurement for Leduc and compares your method with the baseline on the same evaluated games under the same budget.

- **Games.** Validation is the original Leduc game.
- **Data.** There are no training instances. Validation is for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable model state or learned preprocessing on validation. Each game is solved independently through the counted interface during scoring.
- **Budget.** One sweep is two single-player traversals. The combined cap is 1,000,000 sweeps, or 2,000,000 traversals across both runs.
- **Baseline.** AsymP-DGDA with eta=0.1 and mu=0.0001, the Table 4 setting for Leduc, starting from the uniform strategy. It runs 500,000 alternating sweeps in each of the two asymmetric runs. The graph is in `code/efg/algorithm/asymp_dgda.py` and the worker in `code/policy.py`.
- **Measured baseline on validation at 16 CPUs.** NashConv 0.0013944858519354675, std 0 (one deterministic seed). `train.sh` takes 0.32 s and `predict.sh` takes 1.27 s because it only packages the artifact. The solve runs during scoring and takes 1,144 s, about 19 minutes, with the two runs concurrent.

## Metric and guards

The score is last-iterate combined NashConv, lower is better: the sum of both players' exploitability in the evaluated game. Player 1's strategy comes from run 0 and player 2's from run 1. In the baseline, each run perturbs the player it reports. The trusted scorer recomputes NashConv with numerical LiteEFG. A failed scorer guard gives an infinite NashConv.

| Guard | Requirement | Enforcement |
|---|---|---|
| Traversal budget | At most 2,000,000 counted traversals per instance, summed over both run workers and all graph handles. Only this sum is limited; `steps_per_run` is not a per-run limit. An over-budget call fails the evaluation. | scorer |
| Final export | Each worker's output equals the JSON from its latest `export`, and no update follows that export | scorer |
| Valid policy | At every information set, probabilities are finite, lie in [0,1] and sum to 1 within 1e-10 | scorer |
| Protocol errors | Any rejected oracle call invalidates the run, even if candidate code catches the error | scorer |
| Artifact format | `validate.py` checks the public manifest and the sealed ZIP without running code. Policy CSV files are rejected. | validator |
| Worker isolation | Each worker has no network and no filesystem shared with another worker | sandbox |
| Last iterate | The exported policy is the solver's final current strategy, not an earlier, best or averaged one. The service only checks that the declared node is read and that no update follows the export. | patch audit |
| Information access | No hidden game data or seed lookup, and no hard-coded outputs | patch audit |

## Policy interface

Game files stay in the trusted scorer, and submitted algorithms run through a counted service. `train.sh --out MODEL` saves solver configuration. `predict.sh --model MODEL --input INPUT --output OUTPUT` exports a versioned code/model/dependency ZIP artifact. The public input only contains protocol metadata and opaque instance IDs. `data/val/labels/` contains the public validation game for evaluation and method, hyperparameter or checkpoint selection only.

The scorer starts a fresh `code/policy.sh --model MODEL --input INPUT --output OUTPUT` worker for each instance/run. Worker input is JSON with `id`, `run_id` (0 or 1), and `steps_per_run`. Private game text and construction metadata are absent. Worker output is the JSON result of the trusted strategy export. Workers cannot share state, so a method built on one joint run must repeat it in each worker within the combined cap.

Define graphs using `import graph_ir as LiteEFG`, available in `BENCH_BENCH_DIR`, and subclass its `Graph`. It records declarative graph instructions. The scorer validates and interprets these as native LiteEFG numeric operations; it never imports candidate Python or unpickles a candidate graph. The graph API supports forward/backward contexts, colors, constants, graph attributes, arithmetic/comparison operators, aggregation, dot products, projections, normalization, reductions and in-place updates. See `graph_ir.py` for the complete whitelist. The other solvers in `code/efg/algorithm/` still import native LiteEFG, so they need the recorder, and explicit players where missing, before they run here.

Use `leduc_env.open_run(input_path, id, graph, run_id, policy=graph.u)`. The returned oracle provides:

- `update(policy, upd_player=1_or_2, upd_color=[...])`: one counted original Enumerate traversal. An explicit player is required, and other traversal types are rejected.
- `update_many([(policy, player, colors), ...], repeats=n)`: repeat an already chosen bounded schedule. Every native traversal is counted; batching only reduces RPC overhead.
- `snapshot(policy)`: read the declared current finite simplex policy and counters for adaptive scheduling, without adding a traversal. Other graph nodes, including private utilities, cannot be read.
- `export(policy)`: read the current finite simplex policy and trusted cumulative counters. Later updates invalidate that receipt and require a new final export; the output file must equal the latest trusted export. Independent graphs may not be used to substitute an earlier/best/averaged strategy for the required last iterate.

A worker may open multiple independent graphs with `open_run`; each gets a distinct handle and all traversals count against the same original per-instance budget. Every graph declares its output policy, which may be a node or a pair of nodes for the two players. Updates may use another node or the native list-of-strategies form; only the declared output policy may be read. Graph handles never reset counters.

## Interface scope and resources

- **Graph vocabulary.** It includes arbitrary finite literals, scalar/symbolic dimensions, int32 graph colors, random uniform/normal/exponential nodes, and `set_seed`. No separate 4,096-instruction/dimension or 256-color/list cap is imposed. Large instruction streams are uploaded in bounded RPC chunks.
- **Outside the interface.** Create graph nodes through numeric operations such as `const` and `copy`; direct unbound native `GraphNode()` construction is outside the verified interface, and so are arbitrary Python callbacks.
- **Static operations.** They may be declared before or after dynamic operations, but cannot read private utility/reach state without a counted traversal.
- **Rejections.** The service rejects unknown operations, over-budget traversals, raw state exports, non-policy reads and invalid probabilities, including errors caught by candidate code. Each worker may make at most 5,000,000 RPC calls.
- **Resources.** The task uses at most 16 CPUs and has an 8 GB memory setting. Independent runs execute concurrently in separate trusted processes. Native operations run under the overall evaluation deadline and a per-process 8 GiB address-space guard.

## Research contract

**Objective:** Improve last-iterate equilibrium approximation for the fixed Leduc zero-sum games supplied for evaluation, using the counted solver interface.

**Fixed research objects:**

- The original information-set/action tree, legal strategy simplexes, actual zero-sum payoffs and supplied evaluation instances remain fixed; computational regularization does not redefine these game targets. The metric is NashConv in the game itself, so changing an algorithmic regularizer does not authorize replacing the game being solved.
- The original NashConv and the final current policies combined as player 1 from run 0 and player 2 from run 1 remain fixed. Last-iterate convergence is what the paper studies.
- The aggregate limit of 1000000 sweeps / 2000000 counted single-player traversals remains fixed. Figure 4 compares methods at equal update cost, summed across runs.

**Editable methods:**

- Game solvers, auxiliary regularizers, numeric graph structure, permitted stochastic updates, optimizers, regularizer schedules and hyperparameters other than the step size may change within the verified graph interface. Graph formulas and auxiliary optimization objectives may change too; the recorder is not a fixed baseline template.
- The baseline asymmetric DGDA formulas and mu are methods, not mandatory target games, and the paper tunes mu per algorithm and game. The paper also tunes eta, but this task holds it at the Leduc value 0.1 so that a gain comes from the method and not from a better step size. Generic adaptation to counted updates and declared-policy snapshots is permitted.

**Forbidden substitutions:** The step size is fixed at eta=0.1 for every player and sweep: do not change, tune, schedule or adapt it, and do not emulate another step size by rescaling gradients, payoffs or updates.

- Do not modify game payoffs/information sets, report an equilibrium for an easier game, replace NashConv or use an earlier/best/averaged strategy instead of the required final current strategy.
- No hidden game data or seed lookup, uncounted utility reads, raw private graph exports, forged counters or state sharing between independent run workers.
- No training or fitting of reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
