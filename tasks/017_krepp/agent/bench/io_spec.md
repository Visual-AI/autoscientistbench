# Phylogenetic placement I/O

Input is a directory containing `reads.fq.gz` and `manifest.json`. The FASTQ contains existing 150bp reads with independent opaque identifiers and unchanged sequences and qualities. Manifest gives the read count, edge count and root edge. Shared fixed reference index and the global exclusion set are under `$BENCH_DATA_DIR/train`.

Output is UTF-8 TSV with the exact header `read_id\tedge`, then exactly one row for every input read. `edge` is an integer index in the fixed tree postorder numbering; -1 denotes unplaced. Root placement is allowed as a status but excluded from coverage and the mean. Duplicate, missing, extra, malformed or out-of-range values are invalid.

Labels are separate: opaque read-to-query membership, unit-edge errors indexed by query and candidate edge, and immutable per-query baseline placement rates. Validation and test query groups are disjoint. IDs do not encode sequence origin or attachment edge.

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
