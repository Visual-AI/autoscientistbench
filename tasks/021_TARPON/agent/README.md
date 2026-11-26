# TARPON telomeric boundary identification

## 1. Goal
Identify the VRR-region start in human nanopore reads entering the TARPON boundary step. Maximize the identification rate subject to unchanged input populations and per-sample WT+1N repeat purity. VRR means variant repeat-rich: the transition from subtelomeric sequence into the telomeric region enriched in variant repeats. WT is GGTTAG. For this metric, WT+1N counts WT and repeats with at most one nucleotide substitution.

## 2. Data
The original upstream capture/strand/subtelomere filtering is frozen as observed BAM inputs. Every supplied boundary-stage read is retained as an input; no previous boundary decisions or predictions are input. Validation uses both WB60 libraries. BAM sequence/quality and observed XS strand remain, other tags/header identifiers are removed and read IDs are opaque. data/val/labels.json duplicates observations for independent composition checking. There is no training split. Validation is only for evaluation and method, hyperparameter or checkpoint selection; do not fit reusable models or learned preprocessing on it. Inference at prediction time may use the reads in its own input.

## 3. Metric and scoring
Identification rate is retained reads / all subtelomere-pass reads, averaged equally across datasets in the current split. Frozen scorer recomputes repeat composition from submitted boundaries and original observed sequences. Every dataset must preserve its original denominator and mean WT+1N purity must be >= fixed original baseline minus 0.5 percentage points, otherwise score 0. The purity mean gives each retained read equal weight. In each dataset, retained reads with tails shorter than 1000 bases count only up to the baseline's number of such reads, and a median tail below 0.9 times the baseline's median tail gives score 0. Public validation labels hold both baseline values as `short_tail_baseline` and `median_tail_baseline`. Public validation purity baselines are SE_WB60: 97.037 percent and DE_WB60: 97.259 percent. The validator checks sample/read identities and boundary bounds; the scorer also enforces denominators and purity. These guards do not establish biological boundary accuracy. Measured validation identification rate is 0.885316892523603, with standard deviation 0.0.

## 4. Baseline
TARPON source commit e65729b8e0bde173f0c882efb783f70aea2cb3e2. code/identify_telo_start.py and code/variantRepeatAnalysis.py run with the supplied parameters. The baseline scans each read for a sustained rise in WT+1N repeats, then filters low-purity tails and repeat-rich sequence before the estimated start. Upstream preprocessing is frozen; the boundary and composition steps run directly. Existing source behavior, including consecutive_repeats reassignment, is retained. Measured no-op training takes 0.06 s and validation prediction takes 72.54 s.

## 5. Interface
bash code/train.sh --out DIR creates a ready marker. bash code/predict.sh --model DIR --input PATH --output FILE runs boundary and variant analysis. See bench/io_spec.md. BENCH_CPUS sets budget; the source algorithm is deterministic. Python3.12,regex,pysam and native dependencies are copied in the immutable runtime; env.lock stays fixed.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Edit code/; frozen data/,bench/,README and contract remain unchanged. Use all supplied observations with no external data/network. The allocation is up to 12 CPUs. Submit boundaries and filtering status, not fabricated composition tables. Judge independently verifies identities and composition.

## 7. Known pitfalls
The task starts at the fixed boundary-stage input. The baseline runs both supplied Python scripts; alternative boundary methods may replace them. Purity alone is not ground-truth boundary accuracy.

## Research scope

**Objective.** Choose boundaries and retention decisions for the fixed observed human-read population, maximizing identification rate under original composition constraints.

**Fixed research objects.** upstream capture/strand/subtelomere selection and every supplied observed read; boundary-coordinate semantics, per-dataset denominators, original composition computation and purity floors.

**Editable methods.** boundary estimators, filtering decisions and computational/statistical methods on the supplied reads; implementation and internal summaries with honest output boundaries.

**Forbidden substitutions.** When retaining the supplied boundary scripts, changing only sliding_window=100, sliding_window_interval=15, upper_threshold=0.6, lower_threshold=0.05, telomeric_rep_perc=0.7, consecutive_threshold=15, pre_telomeric_repeat_percentage=0.1, the 2000-base upstream span or minimum_telomere_length=0 does not count as a method improvement. Also forbidden: changing the upstream input population, sequences, composition definition or sample weights; submitting fabricated purity summaries, dropping difficult reads from the denominator or using cached decisions. The predicted start must estimate the transition into the variant repeat-rich region. Moving it into a short distal repeat tract only to raise tail purity does not count as a method improvement. The scorer also enforces two limits per dataset: retained reads with tails shorter than 1000 bases count only up to the baseline's number of such reads, and the median tail must be at least 0.9 times the baseline's median tail.

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
