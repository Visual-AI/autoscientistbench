# LexicMap alignment output

Input is a ZIP containing `manifest.json` and FASTQ members. Manifest batches provide opaque read IDs, reference assembly identifier, reference contig lengths, and a FASTQ filename. FASTQ descriptions contain only opaque read IDs. Input supplies no source strand, source interval, or alignment truth. Sequences and qualities are unchanged existing benchmark reads.

Output is a single JSON object with exactly `version` (1), `input_sha256` (SHA256 of input ZIP), `query_ids` (all manifest read IDs in order, including unaligned reads), `hsps` (one `[query_id, subject_id, sstart, send]` row per emitted HSP), and `resources`. The two coordinates are the ends of one base-level alignment, and a row's span, abs(send-sstart)+1, may not exceed 2,000 bp. Coordinates are one-based inclusive, both orientations accepted. Unknown IDs/subjects, missing/duplicate IDs, wrong shapes/types, nonfinite values, duplicate JSON keys, and out-of-range coordinates are invalid.

`resources` requires nonnegative `index_wall_s`, `search_wall_s`, `compile_wall_s`; nonnegative integer `index_bytes` measured using `du -sb` summed over all indexes; integer `cpus` and `seed`; 64-character lowercase hexadecimal `source_sha256` and `binary_sha256`; exact `indexed_groups` and `searched_batches` lists matching the manifest. The baseline's full training builds all ten reference indexes. Full scoring permits up to 32 CPUs. Smoke explicitly builds one reference and uses two CPUs.

Every HSP is scored using the authors' predicate: correct subject and at least one subject endpoint inside the true inclusive interval. A containing HSP with both endpoints outside is not a hit. Off-target counts include every failed HSP in the four hard strata. All six identity strata are scored, including the two controls. Source-location labels live separately.

Resource reports, artifact hashes, and HSP provenance are candidate-reported. Numeric thresholds are checked by the scorer; honest measurement, correct indexing/searching, complete HSP output, actual use of candidate code and reported filtering require patch audit. A plausible numeric report is not independent attestation.

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
