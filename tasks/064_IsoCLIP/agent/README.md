# IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment

## 1. Goal

Improve image-to-image retrieval. The supplied OpenAI CLIP ViT-B/32 encoder is the fixed pretrained research object. Given anonymous
query and gallery image collections, return a similarity for every query/gallery pair.
The objective is `retrieval_map` (higher is better), the equal-weight mean of collection mAP
percentages (theoretical maximum 100). The baseline is training-free IsoCLIP.

## 2. Data

- `data/train/` contains an explanation only: there is no training split.
- `data/weights/ViT-B-32.pt` is the fixed OpenAI pretrained checkpoint; its encoder weights and buffers remain frozen.
- `data/val/input/` contains one packed Caltech101 retrieval instance: 1649 original CoOp validation
  queries and 4128 train-gallery images. `data/val/labels.npz` contains public class labels and
  analysis metadata. Public permutation seed: 20261030.
- Each input NPZ packs encoded image bytes and query/gallery indices. IDs and image order are
  anonymous; identifying image metadata has been removed without changing decoded pixels.
- Hidden evaluation collections are unavailable in this workspace. Public validation is for
  evaluation and method selection only. It is a tuning proxy, so improvements need to
  generalize across image collections.
- Use only the supplied images and the supplied pretrained checkpoint. No external data, additional
  pretrained models, re-splitting, relabeling or embedding labels in code is allowed.

## 3. Metric and scoring

For each query, rank the full gallery by descending similarity. A gallery image is relevant
when its class label equals the query label. Remove self matches using image identity.
AP is the mean precision at the ranks of relevant images, or zero if none remain.
The released torchmetrics computation uses float32 arithmetic and topk tie handling.
Average AP over queries, then average collection mAPs equally. The score is that mean percentage.

| Guard | Class | Enforcement |
| --- | --- | --- |
| Exact instance keys, complete matrices, float32, finite scores, valid timing receipt | scorer and validator | Invalid output; direct scoring returns Infinity |
| Frozen ViT-B/32 architecture, weights/buffers and preprocessing | isolated encoder probe and patch audit | Protocol worst score 0 |
| No additional training data or pretrained models | patch audit | Protocol worst score 0 |
| No hidden-data access, label memorization, hard-coded answers or falsified time receipts | patch audit + timing probe | Disqualification |

Per-collection scores are diagnostics; there is no additional accuracy floor. A prediction file
cannot prove model/data provenance. Those rules require code review. Development baseline: **82.383156046%**, seed standard deviation **0**, seed 1.
The baseline uses one deterministic seed. PASS requires at least 0.1% relative improvement over
the archived hidden reference, subject to the stated protocol guards.

## 4. Baseline

Official source: https://github.com/simomagi/IsoCLIP at
`84f80c9bfb4f0cf201f96b5840dc36cddfc52993`.
`iso_core.py` contains the unchanged `apply_iso` and `encode_image_noproj_openai` functions.
The baseline removes the top 150 and bottom 50 singular directions of the cross-modal projector
operator, projects image features, normalizes them, and computes cosine similarities.

The adapter reads packed images, uses fixed local weights, and reads seeds/threads from
environment variables. Training writes a training-free marker; prediction encodes images
and computes similarities. CLIP is vendored at OpenAI package commit
`551aca347511f782a0f3980e93603194a539e853`. The baseline preserves the encoder, preprocessing,
batch size 32, SVD and hyperparameters. Image batches are encoded by one single-threaded process per CPU; independent
retrieval instances then run in up to four processes with four threads each. No features are precomputed outside evaluation.
Measured with a 16-CPU allocation: no-op train **0.11 s**, development prediction **49.98 s (0.83 min)**. Memory budget: 32 GB; one thread per encoder and up to four threads per retrieval worker.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` creates the model directory and saves all fitted state; the baseline writes a training-free marker.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes one NPZ at exactly FILE.
  See `bench/io_spec.md` for the byte packing and similarity-matrix format.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`.
  Honor the seed whenever using randomness, and never exceed the provided CPU allocation.
- Use `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and
  `bench submit --final` for progress, development evaluation, validation and submission.
- Install dependencies with `uv pip install`; submission freezes `code/env.lock`. Rebuildable
  dependencies use only `name==version` from PyPI or PyTorch CPU. Git/editable/local-path
  dependency declarations are unsupported.

## 6. Rules

- You may replace the post-encoder projector, feature postprocessing and retrieval scoring under `code/`. Keep the CLIP architecture, supplied encoder weights/buffers and image preprocessing fixed. Validation labels may guide selection; they must not be used as query answers or embedded classification tables.
- Frozen surfaces: `data/`, `bench/`, this README, script signatures and prediction format.
- No network during train or predict. No extra images, labels, training data or pretrained models.
  Prediction must use only the supplied input collections and allowed fixed resources.
- Resource budget: 16 CPUs, 32 GB memory.
- One judge seed is used because the baseline is deterministic. Stochastic changes must honor
  `BENCH_SEED`. Reading or guessing hidden examples, hard-coding predictions or paper results,
  falsifying timings, and bypassing the frozen evaluation rules are disqualifying.

## 7. Known pitfalls

- Every prediction starts from raw images. Do not rely on persistent feature caches or network downloads.
- A collection's gallery can differ from its query set. Never assume diagonal self matching;
  use the provided image-index identity relation. The scorer handles self exclusion.
- Preserve float32 scores, every query/gallery pair, and their input ordering. Ties follow the
  released topk operation and can be sensitive to tiny floating-point differences.
- Write NPZ through an open file or ZIP writer: NumPy otherwise appends an unwanted extension.
- `PYTHONSAFEPATH=1` is enabled. Preserve explicit local and vendor `PYTHONPATH` setup.
- Large similarity matrices are intentional. Keep intermediate arrays in local temporary storage,
  and budget image decoding, model loading, output I/O and metric computation in your evaluation.
- Timing receipts are diagnostic and must remain truthful. Model and data provenance require patch audit.

## Research contract

**Objective.** Improve image-to-image retrieval by postprocessing the fixed pretrained CLIP ViT-B/32 representation. CLIP aligns images with text, but its projectors can distort similarities between images. Compare candidate retrieval methods with training-free IsoCLIP on the same fixed representation and evaluation protocol.

**Fixed research objects.** The supplied CLIP ViT-B/32 encoder architecture, pretrained weights/buffers and released image decoding/RGB/resize/crop/normalization are fixed. The supplied raw query/gallery images, self-match exclusion, full-gallery float32 AP/tie handling and equal-collection mean mAP remain fixed. These controls isolate the effect of post-encoder retrieval methods.

**Editable methods.** The projector decomposition/filtering, feature postprocessing, similarity function and retrieval method may change on top of the fixed representation, subject to the cutoff limit under Forbidden substitutions. Validation labels may select configurations but cannot become answers or lookup tables.

**Forbidden substitutions.** Do not replace or fine-tune the encoder, introduce another pretrained model or extra training data, change encoder preprocessing, use answer lookups, change gallery membership/self-exclusion or alter the AP metric. When retaining the supplied IsoCLIP projector, changing only iso_ktop=150 or iso_kbottom=50 does not count as a method improvement.

## Encoder interface

Expose `code/encoder_api.py` with `load_encoder(model_dir, checkpoint)` returning the fixed encoder and image transform, and `encode(model, images)` returning its preprojection image features. The baseline retrieval path uses this same interface. Run `python bench/check_model.py MODEL --checkpoint data/weights/ViT-B-32.pt --code code` in the candidate execution sandbox. The protected probe reconstructs the reference architecture from the supplied weights, checks actual module structure and weights/buffers before and after encoding, and compares preprocessing and features on a synthetic image. It does not import candidate code into the private scorer.

The probe is an operator check, not an automatic proof of the submitted retrieval path. Source review must establish that retrieval actually uses the checked encoder, no separate encoder adaptation occurs, and postprocessing obeys the input/data restrictions. Self-reported hashes are insufficient.

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
