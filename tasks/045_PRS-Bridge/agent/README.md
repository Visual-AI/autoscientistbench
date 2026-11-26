# Constructing Genetic Risk Scores with PRS-Bridge

The judge retrains your code from scratch with its own seeds and compares its predictions against an archived baseline.

## 1. Goal

Predict the quantitative phenotype of a plasmode European-ancestry (EUR) individual from genome-wide genotypes. Use the supplied genome-wide association study (GWAS) summary statistics and linkage-disequilibrium (LD) reference. The fixed setting has causal proportion 1e-3 and no negative selection. Polygenic risk scores (PRS) combine effects across genetic variants. Compare methods by how well they transfer summary-level training evidence to individual phenotypes. Improve `mean_r2` (higher is better), the mean out-of-sample squared Pearson correlation after tuning-half candidate selection over 100 random splits. You may change the method within §6.

## 2. Data

- `data/train/sumdat/chrC/hm3_sumdat.txt`: summary statistics from a GWAS of 100,000 individuals; columns SNP, A1, A2, BETA, SE, N. Chromosomes C=1 through 22.
- `data/train/ref/chrC/`: the supplied block LD matrices and BIM allele/SNP metadata. The supplied SNP and allele metadata define input alignment. The baseline also filters, reconciles and standardizes the training summary statistics before fitting.
- `data/train/variantsC.npz`: SNP and allele order for genotype inputs, arrays `snp,a1,a2`.
- `data/val/input.npz` and `data/val/labels.npz`: a public validation pool of 10,000 individuals. Input contains opaque ids and packed genotypes; labels contain phenotypes and 100 fixed tuning masks. See `bench/io_spec.md` for the packing convention.
- The validation pool has 100 fixed random half/half tuning/scoring splits. Keep the supplied individuals and masks unchanged.
- Fit models and learned preprocessing only on supplied training data. Use validation only for evaluation and method, hyperparameter or checkpoint selection. Do not change the frozen splits, relabel records or add external data. Do not infer private splits from source identities; supplied identifiers are opaque.

## 3. Metric and scoring

For each fixed split, the scorer computes squared Pearson correlation of each of your two candidate PRS columns with phenotype on the tuning half. It selects the higher tuning score (ties choose column 0) and records that candidate's squared correlation on the other half. `mean_r2` is the arithmetic mean of these 100 scores. Both candidate functions must be fixed across the scoring splits. You may duplicate a single predictor into both columns. Correlation R² is invariant to global shift and nonzero scale; it is not `sklearn.metrics.r2_score`.

| Constraint | Enforcement class |
|---|---|
| Exact CSV header, input id order and set, no duplicate/missing rows, two finite numeric columns | scorer pipeline, including validate |
| Candidate selection uses only the tuning half of each fixed split | scorer |
| Correlations must be defined and finite on every half | scorer |
| Each prediction depends only on that individual's genotypes and the model learned from the supplied training data; ids are bookkeeping only | patch audit |
| No hidden-label access, memorized predictions, external data or split inference | patch audit |

An invalid or undefined score is `Infinity` and fails the submission. No additional performance guard is imposed. Predictions must remain consistent when batch composition, row order or opaque ids change, within `atol=2e-5, rtol=2e-5`. This information boundary is enforced by patch audit.

Measured validation `mean_r2`: **0.192989091486**, standard deviation **0.000602015108** across three training seeds. The minimum relative improvement delta is **1%**. The judge runs three training seeds and compares mean scores. PASS requires relative improvement at least `max(delta, 2 × reference standard deviation / abs(reference mean))`; current task values are available through `bench status`.

## 4. Baseline

Official code: https://github.com/YuzhengDun1999/PRSBridge at commit `5e6ae5636189bc7b71f4d264584125ea894db7ea`. `code/PRSBridge.py` implements projection and chromosome-wise fitting; `code/bayesbridge/prsbridge.py` and the prior/coefficient-sampler modules implement the bridge-prior Gibbs sampler with prior-preconditioned conjugate gradients.

The baseline uses two settings: alpha 0.125 or 0.25, percent 0.8, h2=0, h2_se=0, method cg. Alpha controls shrinkage sparsity. Percent is the minimum fraction of LD eigenvectors, smallest eigenvalues first, removed in each block before fitting; eigenvalues below 0.01 are also removed. Summary effects are projected onto the retained eigenspace. Sampled SNP coefficients are averaged to produce additive genotype scores. There are 44 independent chromosome/candidate fits. Actual source iteration semantics are 1000 total iterations with 600 burn-in for alpha 0.125, and 800 total with 500 burn-in for alpha 0.25. They are not added together.

Cython extensions are rebuilt from committed sources in the model output directory. Python 3.11 stdlib distutils provides a fallback when setuptools is unavailable.

Measured on **16 CPUs** across reference repetitions: training took **1131.86, 1096.41, 1095.08 seconds**. Validation prediction took **141.33, 140.38, 132.12 seconds**. The memory budget is **64 GB**, not a measured peak.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train and save everything needed for prediction in DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write a CSV prediction table with exactly `id,pred_0,pred_1`, preserving input order.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Honor the supplied seed and CPU budget. Paths may be absolute.
- Development commands: `bench status`, `bench eval-dev --seed S`, `bench validate-output FILE`, `bench submit --final -m MESSAGE`.
- Dependencies: use `uv pip install` in the venv; submission freezes `code/env.lock`. Python 3.11 is fixed. Only exact PyPI `name==version` pins can be rebuilt; no editable, git or local requirements. Baseline uses numpy 1.26.4, scipy 1.11.4, pandas 2.2.2 and Cython 0.29.37.

## 6. Rules

- You may edit everything under `code/`: predictive model, surrogate loss, internal genotype representation and fitting procedures, with hyperparameters retuned as part of such a method change, while preserving the quantitative-phenotype target, supplied SNP/allele data, interfaces and output format.
- `data/`, `bench/`, this README, the script signatures and output format are frozen.
- Judge resources: 16 CPUs and 64 GB memory. Training and prediction have no network access.
- The judge retrains with three seeds. Every run must honor `BENCH_SEED`; variance affects the PASS threshold.
- Candidate functions must be learned using the supplied training data. Validation labels may support model selection only. Prediction must work for arbitrary input ordering and opaque ids, without using other individuals in its prediction batch.
- Reading or guessing hidden labels/splits, hard-coding predictions, memorizing paper outputs or obtaining outside data disqualifies the submission.

## 7. Known pitfalls

- Genotypes use two-bit PLINK codes, not raw dosage bytes; missing code 01 is mapped to zero dosage by the baseline. The last byte may contain unused bits.
- Keep chromosome SNP and allele order aligned with `variantsC.npz`. The baseline predictor compares the coefficient table's second column to the panel alleles; preserve that convention when checking baseline parity.
- The two output columns are candidate scores, not per-split selected scores. Selection is performed by the frozen scorer. Never choose a column using scoring-half labels.
- The source's `n_iter` already includes burn-in. Increasing it by burn-in changes the baseline training loop.
- `PYTHONSAFEPATH=1` is enabled; entry scripts export `PYTHONPATH`. Build compiled extensions from the committed sources; do not rely on untracked local binaries.
- Parallel chromosome fits share the 16 CPU budget. The baseline gives each fit one BLAS thread to avoid oversubscription.
- Predictions are CSV text regardless of output extension. Supply every id exactly once and in input order.
- Correlation of a constant score is undefined and rejected. Duplicating a nonconstant predictor across the two candidate columns is valid.

## Research contract

**Objective:** Predict the fixed quantitative phenotype from supplied genotype, GWAS summary statistics and LD data, maximizing scorer-selected out-of-sample correlation R^2.

**Fixed research objects:** The phenotype, plasmode EUR study population, supplied reconciled SNP/allele/genotype/GWAS/LD observations, fixed splits and 100 tuning/scoring half-splits remain fixed. Two candidate score functions are fixed across scoring splits; tuning-half selection, squared Pearson correlation and mean aggregation remain fixed.

**Editable methods:** Predictive architecture, shrinkage/prior used as a predictive modeling choice, surrogate training loss, optimizer, numerical method and internal genotype features may change. PRS-Bridge posterior sampling is a baseline predictor, not a fixed posterior distribution being evaluated in this task.

**Forbidden substitutions:** Do not replace genotype inputs or phenotype/allele meaning, import extra cohorts/labels, change splits or target sklearn R^2 instead of the frozen correlation metric. No scoring-half label selection, hidden-ID lookup or predictions depending on other individuals in the prediction batch. When retaining the supplied PRS-Bridge pipeline, changing only alpha=0.125 or 0.25, percent=0.8, h2=0, h2_se=0, or the iteration counts 1000/600 and 800/500 does not count as a method improvement.

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
