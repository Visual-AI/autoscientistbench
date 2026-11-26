# LexicMap existing-query alignment benchmark

## 1. Goal

Research question: how many diverged 1,000 bp reads can a method align back to the correct place in their source genome, without adding wrong alignments and within an index-size budget?

The source paper ("Efficient sequence alignment against millions of prokaryotic genomes with LexicMap", Nature Biotechnology) introduces LexicMap. It samples each genome with 20,000 probe k-mers, finds variable-length prefix or suffix matches (anchors) between query and genome seeds, chains them, and aligns the chained regions at base level.

This task is the 1,000 bp panel of Fig. 3, "Robustness to sequence divergence". That experiment simulates reads from ten bacterial genomes at fixed identities. Each tool builds one index per genome and aligns each read back to its source genome. Tools are compared by the proportion of reads aligned at their source location. The paper compares LexicMap with BLASTn (word sizes 28 and 15), MMseqs2, Minimap2 and Ropebwt3. For 1,000 bp reads at lower identities it ranks BLASTn with word size 15 highest, then MMseqs2, Ropebwt3, Minimap2, LexicMap and default BLASTn.

Here a candidate method is compared with LexicMap v0.7.0 at default settings on the same fixed reads. Improve the mean pooled alignment rate at identities 80, 82, 84 and 86 percent. Identities 95 and 100 are controls. The paper measures speed and scaling to large databases in a separate experiment; they do not enter this score. This is a computational alignment task over fixed reads and reference genomes. It involves no new sequence generation or biological design. Candidate changes are compiled from source.

## 2. Data

`data/train/manifest.json` and `data/train/genomes/` contain the ten complete reference genomes (2.1 to 6.3 Mb) for index construction. This task has no labelled training split.

`data/val/input.zip` contains the validation queries; `data/val/labels.json` holds their source intervals and identity strata. There are 73,106 validation reads in 12 batches, one per genome and identity, from two of the genomes. Validation is for evaluation and for method, hyperparameter or checkpoint selection. No model or learned preprocessing may be fitted on it.

The reads were simulated with Badread using the paper's settings (random error model; no glitches, junk reads, random reads, chimeras or adapters), at 3x depth instead of the paper's 30x. Each read has a 1,000 bp source interval; reads shorter than 1,000 bp after simulation were removed.

Whole genomes define the query split, so related identity strata from one genome stay together. Hidden queries and their labels are held separately. All existing retained queries remain in the full split union. FASTQ sequence/quality bytes are unchanged; descriptions and IDs are replaced with opaque IDs. Do not regenerate, resplit, augment, or relabel the inputs.

## 3. Metric and scoring

`alignment_rate` is 100 times the fraction of queries with an HSP on the true subject whose start or end lies inside the inclusive true interval. Pool counts per identity, then average identities 80/82/84/86. Violating a scorer guard below sets the score to zero. Invalid output is rejected.

| Guard or rule | Enforcement |
| --- | --- |
| Total `du -sb` index bytes ≤ 113716032 (1.5× the baseline's 75810688) | scorer, on the reported value |
| Off-target HSPs per hard query ≤ 0.14 | scorer, recomputed from HSP rows and labels |
| Identity-95 alignment rate ≥ 99.8910031488979%; identity-100 rate ≥ 99.9% | scorer, recomputed from HSP rows and labels |
| Reported CPU count positive and at most the 32 allocated | scorer |
| Output schema, input digest, query order, coordinate ranges, index and search coverage lists | validator and scorer |
| Honest timers, sizes, source/binary hashes, complete HSP output, actual use of candidate algorithms, and reported filtering and implementation provenance | patch audit; self-reported counters alone are not attestation |

Reported search and index times are not scored. Every run reports all ten references in `indexed_groups` (validator). The baseline builds one index per reference in each training run; each evaluation searches its own genome split. The supplied reference genomes and query partitions remain fixed.

Measured validation baseline at 32 CPUs, one seed: `alignment_rate` mean **72.34593063809245**, std 0. Rates by identity: 44.50 (80), 64.83 (82), 84.29 (84), 95.77 (86), 99.98 (95), 100 (100). Off-target is 0.13503 per hard query, close to its ceiling.

## 4. Baseline

Official source: https://github.com/shenwei356/LexicMap at `76366f04afeed14ed1d7c530d033d29cba417db9`, the revision of LexicMap v0.7.0. Go alignment/index algorithms under `code/lexicmap/` are unchanged. `code/adapter.py` compiles candidate Go code with the supplied Go 1.27.3 (the v0.7.0 release used Go 1.24.2) and builds fresh indexes. It then searches each supplied FASTQ file against its batch's named reference, using original default arguments and CPU scheduling (`max(1, cpus//8)` workers). These defaults are the ones in the paper's experiment: 20,000 probes, k=31, minimum anchor length 15, and 17 when only one anchor pair matches.

Measured at 32 CPUs: train 35.74 s (compile 7.39 s, index 28.16 s); validation predict 18.3 s (search 17.28 s).

## 5. Interface (frozen)
`bash code/train.sh --out DIR` compiles source and builds fresh reference indexes. `bash code/predict.sh --model DIR --input PATH --output FILE` searches the queries and emits the JSON HSP/resource format in `bench/io_spec.md`. `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_SEED` and `BENCH_RUNTIME_DIR` come from the harness. This deterministic method does not use a random training seed; the seed is recorded. Only an explicit `BENCH_SMOKE=1` selects the separate one-reference path in `code/adapter.py`; normal calls use all reference genomes.

System Python uses stdlib only. The runtime supplies read-only Go 1.27.3 and pinned module archives; builds are offline (`GOPROXY=off`, `GOTOOLCHAIN=local`). Dependencies cannot change during the task. `env.lock` is fixed.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

The editable surface is `code/`, preserving script arguments and output schema. Data, scoring, README and runtime snapshots are frozen. Full runs have up to 32 CPUs and 32 GB memory; scorer guards remain as above. Every invocation must execute the submitted method and produce fresh outputs. The baseline compiles Go and builds indexes; alternatives may use other algorithms and representations. Reading hidden labels, hard-coded predictions, replaying earlier outputs, forged counters, and hidden source-coordinate lookup are prohibited (patch audit). Source hash strings do not replace audit.

## 7. Known pitfalls
All unaligned queries must appear in `query_ids`. Coordinate orientation does not alter the inclusive endpoint predicate; an HSP containing the true interval with both endpoints outside is deliberately not counted. All six identity strata are required. Every failed HSP in the hard strata counts as off-target, including extra HSPs of a query that also has a hit; failed HSPs in the control strata do not count. Off-target denominators include unaligned hard queries. Search time is elapsed time across the worker pool, not summed process time; index time covers sequential builds and size collection, excluding compilation.

## Research scope

**Objective.** Align the fixed existing query reads to the supplied reference genomes while preserving accuracy, off-target and index-size constraints.

**Fixed research objects.** Reference/query universe, whole-genome partition and all identity strata; inclusive endpoint matching rule, pooled alignment metric, control floors, off-target ceiling and index-size budget.

**Editable methods.** Anchor matching, chaining, extension, alignment and derived internal representations; filtering, scoring heuristics and computational models within the same observed-query problem.

**Forbidden substitutions.** Seeding stays at the paper's defaults: 20,000 probes, k=31, seed desert limit 100 bp with 50 bp spacing, and minimum anchor lengths 15 and 17 (`-m`, `-k`, `-D`, `-d`, `-p`, `-P`). Do not replace it with denser or exhaustive seeding; a gain must come from how anchors are matched, chained, extended and filtered. Do not use the FASTQ quality bytes in seeding, alignment or filtering: the simulator's ideal quality model tracks where the simulated errors are, which real sequence divergence does not provide. Each HSP row must report the two ends of one base-level alignment of the query to that subject, with a span of at most 2,000 bp (span checked by the scorer, alignment by patch audit). Modifying or replacing reference/query sequences or their partitions; changing endpoint correctness, hiding unmatched queries, using origin-label lookups or exceeding the index-size budget; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
