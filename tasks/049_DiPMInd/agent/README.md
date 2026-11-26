# DiPMInd: Distance profile based mutual independence testing for random objects

## 1. Goal
Research question: can a valid test of independence between two random objects, seen only through their pairwise distances, reach higher power than the paper's prod-AD test while keeping type-I error controlled?

The experiment is Example 1 (Section 3, Figure 1) of the paper. Random networks are paired with Gaussian vectors in two models at six dependence levels rho. The paper compares its distance-profile tests with dCov and dHSIC by empirical power, on the same design, metrics and level. Here a candidate test is compared with the prod-AD permutation test.

Improve the guarded mean empirical power of mutual-independence testing from the observed distance matrices. Higher mean_power is better; improvement delta is 0.005.

## 2. Data
The task is synthetic and has no population-level training data; `data/train/` holds only a note. Full validation (`data/val/input/`) contains 2400 independent observed distance pairs: 200 replications for each of two models and rho in {0, 0.16, 0.32, 0.48, 0.64, 0.80}. Every case has n=100 samples, and networks have 200 nodes.

In both models a latent Gaussian vector, equicorrelated at rho on part of its coordinates, drives each network's preferential-attachment power and supplies the Gaussian vector. Each case is an R list with Dx and Dy, two n by n distance matrices: Dx is the L1 distance of the graph Laplacians, Dy the L1 distance of the vectors. The original statistic consumes these two matrices, which are supplied directly after the source's L1 observation transformation.

Inputs expose neither model/rho labels nor latent variables or generator seeds; IDs are independently random and shuffled. Labels are kept separate, in `data/val/labels.json` for validation. Validation generation uses root seed 191001. The frozen generator is in bench/generator.R. Do not re-split or substitute observations.

## 3. Metric and scoring
For every cell, rejection rate is the fraction of its 200 p-values <=0.05. mean_power is the equal mean of the ten alternative-cell rejection rates (rho 0.16 to 0.80 in both models). Size is the rejection rate of a rho=0 cell.

Guards, by how they are enforced:
- Scorer: the score is zero unless size(Model 2,rho=0)<=0.081 and size(Model 1,rho=0)<=0.10.
- Scorer: the complete cell grid and replication counts, exact case matching, and finite p-values in [0,1].
- Validator: columns exactly `case_id,p_value`, one row per input case, finite p-values in [0,1].
- Patch audit: honest use of each case's observed distances and avoidance of known-label shortcuts. The prediction CSV alone cannot prove them.

There are no additional model artifact guards.

## 4. Baseline
Source repository: https://github.com/paromita1991/DiPMInd-code, commit 7473344c22aa0f1fd31215fd4f3fb8063b233d24. The authors' src/teststatRcppsorted_fastest.cpp and the source reproduction's perm_pvalues routine are unchanged. train compiles the original statistic, and predict performs all 500 permutations afresh per dataset, computes all original returned variants, and emits prod-AD. The adapter changes only I/O, compilation location, deterministic permutation RNG and worker control. The source's synthetic generation settings are retained: undirected preferential-attachment networks, m=1, unnormalized graph Laplacians, Model 1 power-interp=sum, original Model 2 median transform.

Measured baseline on full validation, seeds 1,2,3 at 20 CPUs: mean_power mean 0.4263, std 0.0030. size(Model 1,rho=0) is 0.075 and size(Model 2,rho=0) is 0.04 to 0.05. Alternative-cell rejection rates rise with rho from about 0.09 to 0.965 in Model 1 and from about 0.07 to 0.59 in Model 2. train takes about 3 s. Validation prediction takes about 435 s per seed, or about 3.6 CPU-seconds per case. Task resources are 20 CPUs and 80 GB memory.

## 5. Interface (frozen)
Run bash train.sh --out DIR, then bash predict.sh --model DIR --input INPUT_DIR --output FILE, with code as the working directory. bench/io_spec.md defines the CSV/RDS contract. BENCH_DATA_DIR and BENCH_BENCH_DIR expose read-only data and protocol files. BENCH_SEED determines candidate randomness, and BENCH_CPUS limits workers. The copied runtime contains R 4.5.3, Rcpp 1.1.2 and igraph 2.3.4. The system C++ compiler rebuilds candidate source, while Python scoring uses only the standard library. env.lock and the runtime snapshot are fixed: `bench submit` rejects a changed env.lock.

## 6. Rules
The candidate surface is code/. Improvements must preserve the interface, frozen data, scorer and validation schema. The evaluator uses three seeds. Reading held-out labels, embedding cached source predictions, inferring labels from metadata or hard-coding p-values is disqualifying.

## 7. Known pitfalls
- The scorer accepts only the full grid of 200 replications per cell. On a small subset of your own, a single null rejection can break a guard, and the subset cannot establish empirical size or power.
- The graph generation follows the source reproduction's documented interpretation of the paper, including its Model 2 median transform, because the paper does not pin down every generation detail. The paper used igraph 1.3.5. Power on the frozen data need not match Figure 1.
- Model 2 at rho=0 is an exact independence null. Model 1 at rho=0 keeps a weak dependence: given the vector, the spread of the attachment power still depends on the vector's first ten coordinates, through the products V_j V_(j+10). Both size guards apply regardless.

## Research contract

**Objective:** Improve power of mutual-independence tests for the fixed observed random-object distance pairs while retaining empirical type-I control.

**Fixed research objects:** These stay fixed because power comparisons between tests only mean something when every test faces the same null, observations and rejection rule.
- The mutual-independence null, source L1-transformed pairwise-distance observations and n=100 study design are fixed.
- Two model grids, ten alternative power cells, separate null-size thresholds, p<=.05 rejection rule and full replication aggregation remain fixed.

**Editable methods:**
- Test statistics, working dependence models, surrogate fitting losses and valid p-value calibration methods may be replaced for the same mutual-independence null on the supplied distance observations.
- The prod-AD statistic and 500 permutations are baseline methods. B=500 is not a mandatory candidate algorithm or calibration count.
- Numerical implementation and within-case randomization may change using only the same observations.

**Forbidden substitutions:** A p_value must be valid for every law under which the two objects are independent, not only at rho=0: a rejection rule driven by the marginal structure of Dx or Dy alone, including any estimate of rho or of the model, is forbidden. Each case's p_value may depend only on that case's Dx and Dy, artifacts from train.sh and randomization: pooling, ranking, clustering or counting across the cases of an input directory is forbidden. Reject at a nominal level of 0.05: rescaling p-values, taking an uncorrected minimum over several tests or randomizations, or otherwise using the room between 0.05 and the size guards is forbidden.
- Do not redefine the null as a weaker moment-correlation test, substitute raw latent/generated objects, or treat design/rho labels as prediction inputs (patch audit).
- No hidden truth/seed lookup, cached p-values or altered distance inputs (patch audit), and no changed size/power arithmetic (frozen scorer).
- No training or fitting of reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation (patch audit).

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
