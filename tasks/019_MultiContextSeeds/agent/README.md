# Multi-context seeds enable fast and high-accuracy read mapping

## 1. Goal
Map the supplied human Illumina single-end 150 bp reads to the full public CHM13 v2.0 reference using any computational mapping method. Improve mapping accuracy; strobealign-MCS extension alignment is the baseline. The resource allocation permits up to 12 mapping threads.

## 2. Data
- `data/train/reference.fa` is the fixed public human CHM13 reference, with its FASTA index. This is a training-free task; the baseline compiles in the training stage. No training split of reads is supplied. Validation is only for evaluation and selection, never for fitting models or learned preprocessing.
- `data/val/input/` holds the complete existing SIM4 read library and observation-only metadata; `data/val/labels/` holds public origin intervals.
- Complete libraries are held out for evaluation. Read names are opaque stable IDs; sequence and quality bytes are unchanged. IDs cannot be used to join libraries.
- No read simulation, sequence design, download, resplitting, external augmentation, or access to hidden origin truth is part of this protocol.

## 3. Metric and scoring
The metric is `100 * correct_primary_mappings / all_reads`, with same-reference interval overlap as implemented in the authors' evaluation code. Touching endpoints count as overlap, matching that implementation.

The score is the original overlap accuracy. The scorer checks complete label coverage and overlap accuracy; reported runtime is a diagnostic. The validator checks complete observed-read coverage, finite values, coordinate bounds, and the limit of 12 threads. The provenance of reported runtime, integrity of the timer, actual execution of the candidate method, and prohibition on memorized predictions require patch audit. A tool-reported number alone is not trustworthy evidence of runtime.

Measured validation accuracy is 92.4397%, with a standard deviation of 0.0 percentage points across reference repetitions.

## 4. Baseline
The official C++ source is strobealign commit `caa3745fe58ba61dfcebc7e3c419f64b168c94e4`. Mapping logic is in `code/src/`, with the original CMake build. MCS stores long seeds and searches a shorter seed prefix when a full seed match is unavailable. The baseline chains matches and extends them to base-level alignments. Candidate source is compiled afresh, so source edits affect predictions.

Across reference repetitions, compilation took 10.75, 10.85, and 10.65 seconds. Validation prediction took 54.21, 47.59, and 47.86 seconds. Tool-reported validation mapping times were 6.5, 6.3, and 6.3 seconds. These mapping times exclude reference loading, indexing, and wrapper work.

## 5. Interface
- `bash code/train.sh --out DIR`
- `bash code/predict.sh --model DIR --input PATH --output FILE`
- `BENCH_CPUS` is the resource budget; the full protocol uses 12. `BENCH_SEED` is recorded for this deterministic mapper. `BENCH_DATA_DIR` locates the fixed public reference.
- Output details are frozen in `bench/io_spec.md`. No network is available.
- The supplied environment includes compiler/CMake/ISA-L dependencies. Wrappers and scoring use system Python's standard library. `env.lock`, dependencies, and the runtime snapshot are fixed.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Candidate changes belong under `code/`; the script interfaces, data, frozen scoring files, runtime environment, and README remain fixed. Preserve the full input-data and output-coordinate protocol. The mapping algorithm is editable. Source-level method improvements are allowed within these rules.

The allocation is 80 GB memory and up to 12 CPUs. Reading hidden truth, hard-coding alignments, modifying timing provenance, or replaying original outputs is disqualifying.

## 7. Known pitfalls
Validation uses the complete supplied SIM4 library. The tool's reported mapping time excludes reference loading/indexing and output-wrapper work. CMake uses the existing compiler and original `-march=native` setting, so runtime varies by host CPU.

## Research scope

**Objective.** Map the existing human reads to the supplied reference and maximize the original all-read interval-overlap accuracy.

**Fixed research objects.** existing reference and complete read libraries, held-out-library split and coordinate semantics; same-reference inclusive interval-overlap correctness and all-read denominator.

**Editable methods.** mapping/indexing algorithms, alignment scoring and computational representations; implementation and internal heuristics operating on the same observed inputs.

**Forbidden substitutions.** When retaining the supplied mapper, changing only -S=0.5, -M=20, -R=100, -f=0.0002, --mcs=always, -H=50, --gd=0.1, --gl=0.05, --vp=0.7, --sg=10000, --mw=0.01, -A=2, -B=8, -O=12, -E=1, -L=10 or --nams off does not count as a method improvement. Also forbidden: changing the reference or read/quality bytes, generating replacement reads or reconstructing hidden origin labels; dropping unmapped reads, using origin lookup tables or redefining overlap accuracy. Each mapped interval must be the reference span of that read's own alignment or seed placement. Enlarging intervals to gain overlap credit does not count as a method improvement. The validator rejects any mapped interval longer than 300 bases.

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
