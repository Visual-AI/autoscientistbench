# Solving the Poisson equation using coupled Markov chains

## 1. Goal
Estimate the asymptotic variance of h(x)=x for the AR(1) process X[t]=0.99 X[t−1]+N(0,1), using coupled Markov chains. This is the limiting variance of the centered trajectory average scaled by the square root of its length. Improve the estimation of this fixed quantity, measured by asymptotic inefficiency (lower is better): empirical estimator variance times average transition cost.

## 2. Data
`data/train/README.txt` explains the absence of learned parameters. `data/val/input.csv` contains 1000 independent runs, each identified by `case_id` and four initial positions sampled from N(0,4²). These initialize the two independent pairs of chains in the original estimator. `data/val/labels.json` contains the public scoring target and run identities. `bench/generator.py` materializes observations; the validation generation seed is 171001. Generation seeds and scoring truth are absent from prediction inputs. Further transition randomness must honor `BENCH_SEED`. Do not replace observations, reuse validation answers, or hard-code the known analytical target.

## 3. Metric and scoring
`bench/scorer.py` computes sample variance (M−1 denominator) × mean(single + 2×coupled), for M=1000. If |mean−10000| exceeds 3×sqrt(sample variance/1000), the score is +infinity. Full workload size, finite estimates, unique exact case IDs, nonnegative integral counters, positive cost, and agreement between reported estimator cost and callback-ledger cost are scorer/validator guards.

`bench/runner.R` owns the single-kernel and coupled-kernel callbacks and increments the ledger on every actual call. The scorer derives cost from those counters, ignoring the method's cost for arithmetic. The ledger shares the candidate process/output and is **not authenticated against malicious code**. Exclusive callback use, no counter tampering or forged ledger, correct kernel usage, independent runs, and no constant/memorized answers require **patch audit**. Those audit guards are mandatory; numerical checks alone do not establish them.

Measured validation inefficiency is 187920270249.83203, with a standard deviation of 7051230570.521079 across reference repetitions. All validation repetitions passed the mean guard.

## 4. Baseline
The official [unbiasedpoisson repository](https://github.com/pierrejacob/unbiasedpoisson), commit `d1c86281ffbb78ab673911559cd134b756a4ae62`, implements reservoir UPAVE in `code/R/`. It combines two independent coupled-chain estimates of stationary moments. Reservoir sampling selects states at which to estimate a Poisson solution by summing differences along coupled paths until they meet. The baseline uses k=L=500, m=2500, R=50, x₀=0 and reflection-maximal Gaussian coupling. Here k is the averaging start, L is the chain lag, m is the averaging horizon, R is the number of reservoir atoms per signed measure, and x₀ is the Poisson-solution anchor.

Training only writes a protocol marker. There is no training split for fitting reusable models or learned preprocessing. Validation is for evaluation and method or hyperparameter selection only. Do not fit reusable state on validation. Per-run inference may use that run's supplied initial observations and callback-generated trajectories, without carrying fitted state between runs. Bookkeeping took 0.06 seconds in each reference repetition. Validation prediction took 32.42, 31.07, and 31.42 seconds, respectively.

## 5. Interface (frozen)
`bash code/train.sh --out DIR` creates the protocol marker. `bash code/predict.sh --model DIR --input PATH --output FILE` runs the candidate estimator through the frozen callback runner and writes the documented CSV. Environment variables are `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, and `BENCH_BENCH_DIR`.

`bench status`, `bench eval-dev`, `bench validate-output FILE`, and `bench submit` provide the normal development interface. The supplied R runtime includes the required libraries. Python tooling uses only the system standard library. The Python dependency lock, dependencies and runtime are fixed.

## 6. Rules
Candidate changes are limited to `code/`. Keep script arguments, output schema, observed inputs, AR(1) kernel, initial law, independent-run workload and frozen interfaces intact. The original UPAVE tuning is the baseline setting. Candidate unbiased variance estimators, auxiliary working models/losses and numerical optimization may be replaced through `estimate_one(single_kernel, coupled_kernel, rinit, h)` while obeying the same statistical and counted-cost protocol. Data, scorer, generator, runner, README and runtime are frozen. Simulation must go through the supplied callbacks; bypassing them, manipulating counters, forging output or hard-coding the target disqualifies a submission and requires patch review.

The resource allocation is 16 CPUs and 80 GB memory. There is no network during execution.

## 7. Known pitfalls
Coupled transitions count twice, including calls used to estimate Poisson solutions. Scoring uses sample variance, not population variance. A poor empirical mean makes the objective infinite even when the cost is low. The runtime does not include the estimator package; candidate source files are executed directly. Ledger arithmetic is checked automatically, but callback use, independence and unbiasedness still require patch audit.

## Research contract

**Objective.** Estimate the asymptotic variance v(P,h) of the fixed AR(1) Markov kernel using counted transitions, minimizing estimator variance times transition cost.

**Fixed research objects:** P is the specified AR(1) kernel with coefficient .99, h(x)=x, initial law N(0,16), supplied independent initial states and the frozen single/coupled callback oracle. The unbiased asymptotic-variance estimand, 1000 independent runs, sample variance, actual single+2*coupled transition cost and 3-SE mean guard remain fixed.

**Editable methods:** Unbiased estimation constructions, couplings exposed by the fixed oracle, reservoir constructions, auxiliary regressions or control variates and numerical methods may change, with tuning adjusted as part of such a change. UPAVE k/L/m/R values are baseline computation choices; an auxiliary loss cannot alter P, h or v(P,h).

**Forbidden substitutions:** Do not substitute another Markov kernel/stationary distribution, use the analytic answer as a constant, bypass callback simulation, forge cost or share runs. No altered initial observations, hidden seeds or changed transition-cost accounting. When retaining the supplied reservoir UPAVE routine, changing only k=500, L=500, m=2500, R=50 or x_0=0 does not count as a method improvement. Using closed-form results specific to this AR(1) kernel, such as its analytic Poisson solution or stationary moments, in place of estimation does not count as a method improvement.

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
