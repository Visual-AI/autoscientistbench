# Input and prediction contract

Input is a directory containing manifest.json, assembly.fasta plus its index, and coordinate-sorted mapped.bam plus its index. Manifest contigs map observed assembly identifiers to lengths. All alignments and base sequences are existing observations; query names are anonymous identifiers. No hidden strain genome, truth call, or simulator metadata is part of input.

predict.sh receives --model DIR --input DIRECTORY --output FILE. The JSON prediction has exactly version (1), contigs (the input contig-length mapping), vcf (complete eight-column VCFv4.2 text), and runtime. Runtime has wall_s (finite nonnegative calling wall time), cpus (positive integer), exit_code (integer), seed (integer), and caller_sha256 (64 hex characters). VCF must include all contig declarations, the eight-column header, valid positions, and unique variant keys. Empty call sets are valid predictions. The output is one JSON document with no external paths, apart from the informational ##reference line the baseline's VCF header carries.

Labels are a separate directory with manifest.json, few.vcf and many.vcf. They are existing authors-generated truth calls partitioned by the same contigs; no prediction is stored there. The exact authors function counts equal-length-substitution positions, using the few-call set for recall and the many-call set for precision, and ignores contigs absent from the few-call file. F1 is 2PR/(P+R), or zero when P+R is zero. This is the evaluator's behavior, including its lack of an explicit length >5 filter.

The score is zero for malformed output, cpus >32, or nonzero exit. Runtime metadata and source digests are candidate-reported: their authenticity requires patch audit. The judge separately fixes CPU affinity; reported timings are diagnostic rather than independent proof of resource use.

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
