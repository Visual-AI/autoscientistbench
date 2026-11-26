# krepp phylogenetic read placement

## 1. Goal
Can short reads from anywhere on a genome be placed accurately on a large reference phylogeny when their source genome is not in the reference? krepp answers this without alignment or marker genes. In the paper's experiment, 100 query genomes of varying novelty are removed from the WoL-v1 tree (10,575 leaves), simulated 150bp Illumina reads from them are placed on the remaining backbone, and error is the number of edges between each placement and the query's true position. The paper uses it to show that krepp on genome-wide reads has lower mean edge error than EPA-ng on 16S reads, while placing far more reads.

This task keeps the reference, query genomes, reads and error, and compares a candidate with krepp; 16S placement is not part of it. Place the existing reads with any placement method (krepp is the baseline). Improve mean phylogenetic placement edge error, lower is better, while retaining the required placement coverage. The supplied baseline is training-free; the fixed upstream reference index is an input, while the placement implementation under `code/` is editable.

## 2. Data
`data/train/index` contains the fixed upstream WoL-v1 k29/w35/h14 index and its tree files. `data/train/excluded.txt` is the single global exclusion set applied to every placement run: the 100 query genomes, all excluded at once. `data/train/protocol.json` records the edge count (20,603) and the root edge (20602); edges follow the tree's postorder numbering.

There is no training split: `data/train` holds only the reference and no labelled reads. Validation is for evaluation and method, hyperparameter or checkpoint selection only, and no model may be fitted on it.

`data/val/input` contains 100,000 existing reads from 20 query genomes, with separate `data/val/labels.json`. The hidden split is held by the judge and uses disjoint query genomes. The full benchmark preserves 100 query genomes and 5,000 reads per genome. The reads were simulated with ART's default Illumina error profile. Novelty varies by query, so frozen baseline placement rates on val range from 0.34 to 1.0. All reads and tree nodes use independent opaque identifiers; read membership and attachment errors appear only in labels. No reads or genomes are generated during execution. Do not re-split, relabel, or augment the data.

## 3. Metric and scoring
The metric, `mean_edge_error`, is the exact authors' unit-edge distance, measured after removing the query leaf, preserving its attachment edge and suppressing unifurcations. The mean includes only non-root placed reads. The frozen labels hold the exact distance from each query attachment to every output edge; the scorer averages those integer errors without rounding.

Scorer-enforced guards: overall non-root placement rate must be at least 0.881 on test and 0.858 on val; each query's rate must be at least its frozen source-baseline rate (`baseline_rates` in the labels) minus 0.10; every read must have exactly one status. Root and unplaced statuses do not count as placed. A failed guard receives the mean error of placing every read sister to the root clade, weighted by all input reads (27.25 on val). The floors above are in the scorer's code; it does not read `minimum_overall_rate` in the labels. The validator rejects duplicates, missing or extra reads, non-integer edges, and invalid edges.

Patch-audit constraints: retain the full global reference exclusion set and input protocol; do not recover or memorize query-origin associations, use cached placements, modify the supplied fixed index, or replace the full input with the explicit smoke fixture. Opaque excluded leaves are present to preserve the original index format; they have no input mapping to read origins. These restrictions require patch review in addition to numeric scoring.

## 4. Baseline
The method is krepp v0.11.1 from https://github.com/bo1929/krepp at source commit `1aa143ca173bdd0324e6f8d754ca25e61da0f646`, including existing global leave-out and effective-clade-size changes. The maximum pseudo-likelihood implementation, default placement parameters and numerical routines are preserved. This task's version changes only identifiers, data paths, entry points, candidate compilation and thread control.

For each read, krepp finds k-mer matches within Hamming distance 4 through its locality-sensitive hash index and estimates a distance to each matched reference by maximum pseudo-likelihood, extended to clades. It chooses the largest clade whose distance is statistically tied with the minimum, by a likelihood-ratio test at the 10% level. A read stays unplaced if the chosen clade has too little match evidence at Hamming distance 2 or less, or if that clade is the root. In the paper, placement rates fall from close to 100% for the least novel queries to below 50% for the most novel, error rises at high novelty, and errors have a long tail.

Measured at 16 CPUs over seeds 1, 2 and 3: val mean edge error 1.9080 (std 0.0036) and overall val placement rate 0.8596. A clean compile takes about 18 seconds. Placing the 100,000 val reads takes 239 to 288 seconds. An earlier baseline run peaked at 52.6 GB of memory.

## 5. Interface (frozen)
`bash code/train.sh --out DIR` compiles the current candidate C++ source and places its executable in DIR. `bash code/predict.sh --model DIR --input PATH --output FILE` runs that executable on the supplied input directory. The output is a two-column TSV, one status per input read; details are in `bench/io_spec.md`.

`BENCH_SEED` is recorded but does not affect baseline placement. Repeated baseline runs place the same reads, yet 0.6% to 0.9% of val reads changed edge between runs, which produces the std above. `BENCH_CPUS` controls compilation and placement threads. `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_RUNTIME_DIR` are supplied by the harness. The runtime is a fixed copied snapshot containing the original Python environment and interpreter, native compiler, libraries, and C++ headers. Its dependency lock is immutable: `bench submit` rejects any change to `code/env.lock`, so dependencies cannot change.

## 6. Rules

The research contract below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

The editable surface is `code/`, subject to the protocol and guards above. Data, scorer, validator, I/O specification, README, snapshot and dependency lock are frozen. The allocation is up to 16 CPUs and 80 GB memory. The judge uses seeds 1, 2 and 3. Network access is disabled.

Do not read hidden labels, embed placements or paper results, bypass the scorer, or bypass execution of the submitted method (patch audit). The baseline compiles C++; alternative implementations may use other supported languages and artifacts. The explicitly selected smoke fixture is permitted only when the harness sets `BENCH_SMOKE=1` and is never a substitute for the default data.

## 7. Known pitfalls
The complete fixed index is about 52 GB and loading it dominates small full-index runs. Root placement is a valid status but reduces coverage. Leave-out exclusion operates at matching time and updates effective clade sizes; the supplied index must remain unmodified, but a candidate may construct its own internal representation while honoring the same global exclusion set. Queries can remain unplaced, but all reads still need explicit output rows. Reference trees and read headers must retain their opaque IDs.

## Research contract

**Objective.** Place each observed read on the fixed reference phylogeny, minimizing unit-edge placement error subject to coverage requirements.

**Fixed research objects.** Reference tree/index information and edge numbering, global exclusion set and read/query partition; unit-edge distance after the specified leaf-removal procedure and overall/per-query coverage floors. These stay fixed because the comparison means something only when every method places the same reads on the same backbone, with the query genomes absent, under the same error and coverage terms.

**Editable methods.** Placement likelihoods, inference/search algorithms and derived internal reference representations; computational models and implementation while honoring reference exclusions.

**Forbidden substitutions.** Excluded leaves may only be removed from the reference: do not use their identity or tree position, or the placements of other input reads, as a prior on where a read attaches (patch audit). Changing the reference phylogeny or scored edge geometry, altering observations or admitting excluded reference information; using query-origin tables, memorized placements or omitting unplaced reads; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. The validator rejects omitted reads and the labels freeze the scored edges; everything else here rests on patch audit.

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
