# 019_MultiContextSeeds I/O

`train.sh --out DIR` saves method artifacts; the baseline compiles C++ source into `DIR/strobealign`. `predict.sh --model DIR --input INPUT_DIR --output FILE` runs the submitted method and emits the common coordinate format. The baseline uses MCS extension alignment; alternative mapping algorithms are permitted.

Inputs contain `input.json` and single-end 150 bp `reads.fastq.gz`. Read names are opaque stable IDs. For full inputs the fixed public CHM13 reference is `BENCH_DATA_DIR/train/reference.fa`; the separate explicit smoke fixture supplies an unaltered chrM reference locally. Input directories contain no simulated origin intervals.

Output line 1 is JSON with exactly `schema` (1), `mapping_time_s_tool_reported` (finite nonnegative seconds), `threads` (positive integer), and `seed` (integer). Line 2 is `id<TAB>reference<TAB>start<TAB>end`. A mapped interval may be at most 300 bases long. Remaining rows contain exactly one primary mapping for every input read. Mapped intervals are zero-based reference coordinates with `0 <= start < end <= contig length`; unmapped rows use `*`, `-1`, `-1`. Duplicate, missing, foreign IDs, invalid intervals, or nonfinite runtimes are rejected. The full protocol permits up to 12 threads.

Labels are separate `labels.json` plus `intervals.tsv.gz`. Accuracy is the percentage of all reads whose primary mapping overlaps the origin interval on the same reference, preserving the authors' inclusive endpoint overlap convention. The score is overlap accuracy; reported elapsed time is diagnostic. The scorer uses the same overlap convention in full and smoke modes. Authenticity of timing and actual execution of the submitted method remain auditable.

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
