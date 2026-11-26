# ALPINE long-read outcome classification

## 1. Goal
Classify existing long-read amplicon observations at the human TRAC and TRBC target sites across PacBio HiFi and Oxford Nanopore (ONT) technologies. Improve mean ONT overall read-level F1 while retaining all four pools' integration and overall accuracy constraints. Compare the candidate with the supplied ALPINE classifier on the same observations. Outcomes include unmodified reads, SNPs, small/large insertions and deletions, and donor-specific integration classes. HDR means homology-directed repair; ITR means inverted terminal repeat. Integration classes distinguish HDR, integration with ITRs, truncated integration without ITRs, and cross-target integration from the other donor.

## 2. Data
The supplied baseline is training-free. data/val/input holds observation FASTQs, opaque IDs, and existing reference/config files. The observed sequences and quality bytes are fixed. Labels are separate in data/val/labels.json. Input read names/comments were replaced because original names contained ground truth. No sequence generation or external data is needed. There is no designated training split. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Do not fit models or learned preprocessing on validation observations or labels.

## 3. Metric and scoring
f1 is the mean of the two ONT Total F1 percentages, with per-pool F1 rounded to 2 decimals exactly as the evaluator. Integration uses exact-class TP/FP/FN across the five relevant groups. Every pool must maintain integration F1>=baseline-.2pp and Total F1>=baseline-.5pp; otherwise score 0. Here baseline means the fixed guard reference values in bench/io_spec.md, not the current validation result. The prediction contract is in the same file. All guards are scorer enforced. All supplied reads must be represented; missing/extra/duplicate IDs are rejected. The latest validation reference has mean f1 87.385 and standard deviation 0.0 over one seed.

## 4. Baseline
ALPINE upstream b59121d132f20e5056b76038b50dc136a91a756b, unchanged computational source under code/ALPINE. The CWL per-sample workflow runs filtering, alignment, sorting, classification, counting and output grouping. The old minimap2 used by the classifier's postprocessing is included. Original classification defaults are unchanged. At 16 CPUs, the measured reference took 0.31 seconds for training preparation and 145.72 seconds for validation prediction.

## 5. Interface
bash code/train.sh --out DIR prepares local CWL definitions. bash code/predict.sh --model DIR --input PATH --output FILE executes all four pools and exports read classes. BENCH_CPUS caps aligner threads; BENCH_DATA_DIR/BENCH_RUNTIME_DIR locate data and copied environments. bench eval-dev,bench validate-output and bench submit use standard interfaces. Runtime is fixed copied conda+CWLtool+minimap2; env.lock is immutable.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Only code/ is editable. Frozen bench/,data,README and output format must not change. Reference/config inputs define the classification problem. Do not read labels during prediction or infer classes from original simulation names. No network or original-reproduction access is available. The allocation is up to 16 CPUs and 80 GB memory. Candidate scripts must run; cached predictions are not a method.

## 7. Known pitfalls
Python script import directories must be explicit under PYTHONSAFEPATH. Classification defaults filter low-quality observations; the adapter marks absent output reads Unclassified so they remain scored. Tiny smoke has44 existing reads per pool and may score0 under unchanged full guards.

## Research scope

**Objective.** Classify the existing observed long reads into the fixed outcome classes while improving ONT F1 and retaining all-pool validity.

**Fixed research objects.** Four supplied read pools, reference/configuration definitions and stratified partition; outcome ontology, original rounded F1 aggregation and integration/overall control thresholds. These keep the observations, targets and comparison criteria constant.

**Editable methods.** Computational classification, alignment, filtering and learned prediction methods; training surrogates and implementation with complete fixed read coverage. All changes remain subject to the validation-use rule.

**Forbidden substitutions.** Changing observed sequences, target definitions, outcome categories or scoring population; inferring labels from removed simulator identifiers, reusing cached class outputs or altering the control thresholds. Do not fit reusable model state or learned preprocessing on validation features or labels. When retaining the supplied ALPINE pipeline, changing only variant_window=20, min_itr_length=10, full_hdr_cutoff=0.99 for HiFi or 0.95 for ONT, or min_qual=30 for HiFi or 10 for ONT does not count as a method improvement.

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
