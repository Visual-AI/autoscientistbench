# Retrieval I/O

`--input` is a directory of anonymous retrieval instances, one NPZ per instance.
Load with `allow_pickle=False`. An instance contains:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| images | uint8, total encoded bytes | Concatenated original image streams with identifying metadata removed |
| offsets | int64, image count + 1 | Image i occupies images[offsets[i]:offsets[i+1]] |
| query | int64, Q | Query image indices in the prescribed output row order |
| gallery | int64, G | Gallery image indices in the prescribed output column order |
| rgb_first | bool, image count | Whether the released loader converts RGB before CLIP resizing |

Image order and file IDs are opaque. There are no class labels or source filenames in inputs.
An identical query/gallery image index denotes self retrieval; the scorer removes that pair.
Apply the supplied frozen CLIP transform, honoring rgb_first before resizing as in the released loader. The encoder preprocessing is fixed; feature postprocessing and projector/retrieval methods may change.

`--output` is one NPZ at the exact supplied filename, including when no extension is given.
Each input filename stem is a key containing a finite float32 matrix of shape `(Q,G)`.
Higher values mean greater similarity. Output rows and columns follow query/gallery order.
Duplicate, missing or additional instance keys are invalid. An additional required key `__meta__`
is a scalar JSON string containing `wall_s` (nonnegative finite prediction wall seconds),
`cpus` (the integer 1–16 budget supplied as BENCH_CPUS, not the number of active workers), and optionally `seed`. The receipt must include imports, model loading,
image decoding, encoding, projection, and output writing. It is independently audited using
external stage timing; falsifying it disqualifies the submission.

Labels are a single NPZ: `ID__query` and `ID__gallery` are integer class arrays;
`ID__self` stores the matching gallery index or -1. `__meta__` describes the public validation
instances for analysis. The prediction is independent of label files.

`validate.py PRED INPUT_DIR` checks keys, dimensions, float32 dtype, finite values and receipt.
`scorer.py PRED LABELS` prints one JSON line with `retrieval_map`, `raw_map`, per-instance mAP,
and runtime diagnostics. Full-gallery AP uses the released torchmetrics topk ordering and
float32 precision arithmetic, then averages queries within each instance and instances equally.
Malformed output returns Infinity. Model provenance and data-use constraints need patch review.

## Research scope and encoder probe

Improve image-to-image retrieval by postprocessing the fixed pretrained CLIP ViT-B/32 representation. The supplied CLIP ViT-B/32 encoder architecture, pretrained weights/buffers and released image decoding/RGB/resize/crop/normalization are fixed. The supplied raw query/gallery images, self-match exclusion, full-gallery float32 AP/tie handling and equal-collection mean mAP remain fixed.

The projector decomposition/filtering, feature postprocessing, similarity function and retrieval method may change on top of the fixed representation. Validation labels may select configurations but cannot become answers or lookup tables.

Do not replace or fine-tune the encoder, introduce another pretrained model or extra training data, change encoder preprocessing, use answer lookups, change gallery membership/self-exclusion or alter the AP metric. When retaining the supplied IsoCLIP projector, changing only iso_ktop=150 or iso_kbottom=50 does not count as a method improvement.

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
