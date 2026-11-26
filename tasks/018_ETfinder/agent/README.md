# ETfinder fixed-database host-record benchmark

## 1. Goal
Analyze the released database for the Rhodobacter sphaeroides case. Its observations link RecT records to host SSB sequences, genomic-neighborhood annotations and taxonomy. Select existing host records that agree with curated candidate records, measured by species-level record F1. This evaluates database analysis, not biological activity. No experimental construction or sequence design is part of the task.

## 2. Data
The complete released observations are in data/train/records.json. Original annotations are in data/train/assemblies.dat, data/train/16S/ and readonly data/train/taxonomy/. No curated labels are method inputs. data/val/input.json describes the prediction universe. data/val/labels.json is only for evaluation and method or hyperparameter selection, not fitting models or learned preprocessing. Identity groups keep species aliases and unresolved-name matches together. Validation contains 20% of groups. Existing sequences remain unchanged. Missing RNA caches retain the original missing-data treatment; offline runs never download replacements.

## 3. Metric and scoring
Species-level record F1,higher is better. The scorer preserves the original maximum one-to-one taxid matching with normalized-name fallback for unresolved ranks. Repeated species across distinct records remain separate denominators. Global candidate count must lie in45..182 inclusive,otherwise score0;population projection occurs afterward. Unknown/duplicated assembly IDs are rejected. Frozen taxonomy determines identities. Method/protocol authenticity requires patch review. Reference validation F1 is 0.5217391304347826, with standard deviation 0.0 from one run.

## 4. Baseline
The baseline executes the released CLI and funnel from the supplied observations. The funnel filters host records by the target SSB motif and annotated RecE partners, then keeps one representative per host RNA cluster and every remaining assembly without cached RNA. The adapter emits these assemblies; the separately computed ranking is not used for the output. The training script creates a marker without fitting. The measured marker time was 0.11 seconds, and validation prediction took 2222.5 seconds.

## 5. Interface
bash code/train.sh --out DIR creates a marker for this training-free pipeline. bash code/predict.sh --model DIR --input PATH --output FILE executes the method and emits assembly IDs;see bench/io_spec.md. Honor BENCH_SEED,BENCH_CPUS,BENCH_DATA_DIR,BENCH_BENCH_DIR. Copied Python3.10/native tools are supplied in snapshot mode;env.lock is immutable,no installation/network required.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Only code/is editable;data,bench,I/O and README are frozen. The observed records, identity matching, output universe and scientific score guards remain binding; the released ranking pipeline and funnel are a replaceable baseline. Do not read,embed or reconstruct curated labels from external resources. Do not substitute cached predictions. The allocation is up to 16 CPUs. All outputs must be observed assembly IDs.

## 7. Known pitfalls
The metric counts host records, not deduplicated species. Size guards use the global list even when one population is scored. Tiny smoke can validly score0 through the unchanged guard. The supplied wrapper calls IQ-TREE3. Source imports resolve the current candidate and frozen evaluator helpers. Copied taxonomy is readonly with no automatic updates.

## Research scope

**Objective.** Analyze the existing fixed database and select observed records for comparison with the curated species-level record-matching target.

**Fixed research objects.** Existing database, supplied observations, taxonomy/matching semantics and allowed record-ID universe; record-level species F1, identity-group split and candidate-count validity range.

**Editable methods.** Computational ranking, selection and statistical analysis of the existing records; implementation and internal representations without changing the biological observations.

**Forbidden substitutions.** Adding or modifying biological sequences, replacing the database or redefining taxonomic matches; external reconstruction of curated answers, cached selections or collapsing record-level denominators. When retaining the supplied funnel, changing only the motif LDDEIPF, the RecT length bounds 200 and 700, the three RecE-partner Pfam entries, the host RNA clustering identity 0.99 or its coverage 0.9 does not count as a method improvement.

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
