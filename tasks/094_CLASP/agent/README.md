# CLASP online convex learning

Policies are evaluated through a private causal service using the train, predict, validator and scorer interfaces.

## Experiment and score

Each episode has n=10, k=4 and T=100. Validation contains 100 episodes. The online loss/constraint generator and environment-provided initial points are fixed.

For each round, let v_t = max(max(A_t*x_t-b_t), 0). Episode loss is sum_t 0.5*||H_t*x_t-y_t||^2, CCV1 is sum_t v_t, and CCV2 is sum_t v_t^2. The score is mean episode CCV2, lower is better. Guards require mean loss at most 1.01 times the baseline mean loss on the same episodes and mean CCV1 at most 1.05 times the corresponding baseline mean CCV1. Actions must be finite and lie in [0,1]^10, with a 1e-6 numerical tolerance. Reported metrics must exactly match the trusted finish record; independently recomputed totals use rtol=atol=1e-8. A failed guard or invalid record gives an infinite score.

Relative gain is (reference mean - candidate mean)/abs(reference mean). Passing requires gain >= max(0.05, 2*reference std/abs(reference mean)).

## Development data

There is no offline training split. `data/val/input.json` lists opaque episode IDs. Public validation observations and labels live under `data/val/labels/`. They are for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable models or learned preprocessing on validation. Online updates may use feedback already revealed within their own episode. Policy workers receive one assigned ID and an RPC service, without the outer validation directory.

## Candidate interface

`train.sh --out MODEL` prepares the model. `predict.sh --model MODEL --input INPUT --output OUTPUT` exports a versioned ZIP artifact containing code, model and locked dependencies. It does not run a test episode. Implement `code/policy.sh --model MODEL --input INPUT --output OUTPUT`; it is executed in a new isolated worker for each episode. The input is a JSON object with `id`. `BENCH_BENCH_DIR` contains the public RPC client.

`ep = clasp_env.make_episode(episode_id)` opens the assigned episode once. `initial()` returns the starting point. `step(x)` commits a decision in [0,1]^10 before returning that round's (H,y,A,b); `step_many(xs)` commits the supplied batch sequentially. `rounds` is a trusted counter. After 100 steps call `finish(final_x)`, returning the full trajectory and accumulated metrics.

Write one CSV row with columns `id,actions,rounds,loss,ccv1,ccv2`. The actions field is a JSON array of 101 finite vectors, each with 10 coordinates. The first 100 decisions are scored; the last vector is the final unscored update. The private scorer compares the row with the trusted finish record and rejects forged trajectories or counters. RPC calls on another ID, reset attempts, out-of-order calls, excess rounds and calls after finish are rejected, even if the worker catches the exception. Each episode starts with a fresh worker; algorithm randomness comes from `BENCH_SEED`.

Everything under `code/` remains editable, including the initial decision policy, update rule, hyperparameters and training procedure, subject to the causal and validation-use rules. `initial()` exposes the environment-provided random starting point; custom methods may choose their initial decision. All 100 scored decisions and the final unscored update must be submitted.

## Resource budget

The memory budget is 8 GB. By default, at most eight episodes run concurrently, bounded by the process's actual CPU affinity, with one CPU per worker. To parallelize within an episode, optionally add `code/policy_resources.json` containing `{"episode_cpus": 8}` (an integer from 1 to 8). The scorer assigns nonoverlapping groups of that many CPUs, reducing each group to the actual allocation if fewer CPUs are available. `BENCH_CPUS` and numerical thread defaults reflect the worker's actual group; candidate code may configure its own thread usage within that allocation. The released baseline itself stays single-threaded.

## Validation reference

The validation baseline mean CCV2 is 81.923051, with algorithm-seed standard deviation 0.0 from one run. This does not measure uncertainty across episodes. For algorithm seed 1, training took 0.26 s, validation artifact export took 0.87 s, and the full validation policy replay took 39.35 s. These are measured times, not time limits. Predictions use the executable artifact protocol described in `bench/io_spec.md`; outer trajectory CSV files are invalid.


Decision policies, auxiliary surrogate losses, internal models, update rules, optimizers and hyperparameters may change. The actual per-round quadratic loss, linear constraints, decision domain, causal feedback order and loss/CCV scoring formulas are fixed; changing an internal surrogate does not change those research objectives.

## Research contract

**Objective:** Minimize squared cumulative constraint violation in the fixed online convex-learning environment while satisfying its original loss and CCV1 guards.

**Fixed research objects:** The original episode generators/data with n10/k4/T100, decision domain [0,1]^10, per-round loss .5*||H*x-y||^2 and linear violation max(max(A*x-b),0) are fixed. Each decision is committed before its round feedback; all 100 scored decisions and the final unscored update are required. The supplied data splits and mean sums of violation^2 remain fixed. Mean loss<=1.01*baseline and mean CCV1<=1.05*baseline remain fixed and use independently reconstructed original metrics.

**Editable methods:** Online policies, initial-decision selection, internal predictive models, auxiliary surrogate losses, updates, preconditioners, optimizers and numerical schedules may change. The returned original initial point is legal information; the candidate may choose its own first decision, as in the existing interface. CLASP update equations are a baseline.

**Forbidden substitutions:** Do not replace the true quadratic loss/linear constraints, change the domain/horizon/CCV definitions, expose round feedback before the associated action, or omit difficult rounds. No hidden future parameters/seeds, reset/query-budget evasion, forged trajectories/metrics or cross-episode private state. After the initial action, the policy must adapt its decisions using the episode's revealed loss and constraint feedback; constant actions and preset trajectories that ignore this feedback do not count as a method improvement. Patch audit checks this rule. Changing only the coefficient or exponent of the step size 1/sqrt(t+1) in `clasp.py` also does not count.

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
