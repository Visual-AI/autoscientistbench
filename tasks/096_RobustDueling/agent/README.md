# Robust linear dueling bandits

Policies are evaluated through a private causal service using the train, predict, validator and scorer interfaces.

## Experiment and score

Each episode has T=2000 rounds, pre-serving dimension d=10, post-serving dimension e=20 and K=10 arms. The public validation set contains 10 episodes. Context coordinates are uniform on [-pi, pi]. The fixed mapping is y=[cos(x), sin(x)], and utility is u=x·theta+y·zeta. Each episode draws independent centered Gaussian coefficients with standard deviations 0.1 for theta and 5 for zeta. Clean preferences follow sigmoid(u_a-u_b). The corruption budget is C=25: outcomes agreeing with the utility ordering are flipped until this budget is spent and released immediately. Other outcomes have rounded, nonnegative Gaussian delays with mean and standard deviation 100.

The score is arithmetic mean final cumulative regret, lower is better. Each round contributes half the sum of the gaps from the best arm utility to each selected arm utility. Sum these contributions over the full horizon, then average over episodes. Trusted scoring reconstructs this value from hidden utilities and committed actions. Across evaluation seeds, the relative reduction in mean regret must reach max(0.05, twice the reference standard deviation divided by its absolute mean). Invalid artifacts or incomplete or mismatched trajectories fail evaluation.

## Development data

`data/val/input.json` lists opaque episode IDs. Public validation data and labels live under `data/val/labels/` and are for evaluation and method, hyperparameter or checkpoint selection only. There is no offline training split. Do not fit reusable models or learned preprocessing on validation data. Within an episode, online updates may use its causally released contexts and preference feedback. Discard fitted episode state afterward. Policy workers receive one assigned ID and an RPC service, without the outer validation directory. IDs do not authorize reconstructing private environment state.

## Candidate interface

`train.sh --out MODEL` prepares the model. `predict.sh --model MODEL --input INPUT --output OUTPUT` exports a versioned ZIP artifact containing code, model and locked dependencies. It does not run a test episode. Implement `code/policy.sh --model MODEL --input INPUT --output OUTPUT`; it is executed in a new isolated worker for each episode. The input is a JSON object with `id`. `BENCH_BENCH_DIR` contains the public RPC client.

`ep = dueling_env.make_episode(episode_id)` opens the assigned episode once. Each round calls `get_context()`, `serve(a,b)`, then `advance()`, in that order. Delayed preference feedback is returned only by `advance()`. After all rounds call `finish()`; it returns the committed actions, actual rounds and regret.

Write one CSV row with columns `id,actions,rounds,regret`. The actions field is a JSON array of 2000 integer `[a,b]` arm pairs. The private scorer compares it with the trusted finish record and rejects forged trajectories or counters. RPC calls on another ID, reset attempts, out-of-order calls, excess rounds and calls after finish are rejected, even if the worker catches the exception. Each episode starts with a fresh worker; algorithm randomness comes from `BENCH_SEED`.

You may change the dueling algorithm, internal feature transformations or predictive models, auxiliary learning loss and training procedure, and retune hyperparameters as part of such a change, while preserving the same environment and causal protocol. This is not restricted to the supplied RCDP-UCB baseline.

Treat IDs as opaque handles: do not reconstruct instance randomness or read private seeds, future contexts, utility arrays or unreleased outcomes. Each episode must start with fresh algorithm state. For a fixed `BENCH_SEED`, its output must be independent of input order or which other episodes were requested. Model artifacts may contain legitimately trained parameters, but may not carry observations or fitted state from another evaluation episode. These method and randomness rules still require code review; the causal service checks observable interactions.

## Resource allocation

The total resource quota is 8 CPUs and 16 GB RAM. The default runs at most 8 episodes concurrently, with one numerical CPU thread per worker; fewer CPUs in the enclosing process affinity reduce concurrency.

Candidates may request more CPUs within an episode by adding `code/policy_resources.json` containing only `{"episode_cpus": N}`, where N is an integer from 1 to 8 (default 1). The worker receives at most N CPUs from the actual affinity and the matching `BENCH_CPUS`. For N greater than 1, episodes run sequentially. An algorithm using multiple CPUs should configure its own `policy.sh` accordingly; the supplied baseline keeps numerical threads at 1.

## Validation reference

The baseline validation mean is 9608.3746, with population standard deviation 186.7449 across seeds 1, 2 and 3. For those seeds, training takes [0.31, 0.31, 0.31] seconds, validation artifact export takes [2.83, 2.82, 2.82] seconds, and causal validation replay takes [40.41, 40.08, 40.79] seconds. The outer prediction must be the policy ZIP artifact; trajectory CSV files are only worker outputs.


Dueling policies, internal feature transformations/predictors, auxiliary learning losses and optimizers may change, and hyperparameters may be retuned with such a change. The actual environment utility model, context generator, delay/corruption process, causal service and regret target remain fixed. Methods may compute features from legally revealed contexts; this does not grant access to private utility parameters or unreleased service data.

## Research contract

**Objective:** Minimize cumulative dueling regret in the fixed contextual utility environment with delayed corrupted preferences and the original pre/post-serving feedback order. Compare candidate policies with the supplied RCDP-UCB baseline under the same environment and feedback order.

**Fixed research objects:** The T2000/d10/e20/K10 contexts and utility law, sinusoidal environment feature mapping, private parameter draws, Gaussian delay process and corruption budget25 remain fixed. get_context reveals current pre-serving contexts; serve commits the arm pair and returns selected post-serving contexts; advance reveals only due delayed feedback. The supplied episode split, independent fresh state and mean cumulative regret remain fixed.

**Editable methods:** Any dueling algorithm, internal preference/utility estimators, feature engineering, predictors of post-serving contexts, auxiliary losses and optimizers may change, and hyperparameters may be retuned with such a change. Transforms computed from legally revealed inputs and public problem structure are valid methods, including nonlinear features; the supplied linear RCDP-UCB learner is not a required candidate architecture. Estimating utility parameters from released feedback is permitted.

**Forbidden substitutions:** Do not substitute an easier utility law, context distribution, uncorrupted/instantaneous feedback channel or different regret definition. Do not obtain true utility coefficients through private seed/state lookup, access unreleased outcomes/future contexts, query the service early, forge actions/regret or carry evaluation-fitted state between episodes. When retaining the supplied RCDP-UCB pipeline, changing only lambda_reg=1.0, alpha=0.1, C=25, Lambda=10000, mu_tau=100, kappa=0.25, the mapping Adam lr=1e-3 or the mapping epochs=2 does not count as a method improvement. Replacing the learned post-serving map with its generating closed form `y=[cos(x), sin(x)]` does not count as a method improvement, even though structural transforms are otherwise allowed.

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
