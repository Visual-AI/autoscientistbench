# Fixed-network dropout-allocation contract

Inputs are NPZ tensors with sorted unique integer `ids` and the supplied normalized float32 `x` of shape `(N,3,32,32)`. Training additionally supplies integer labels `y` in0..9. Preserve the provided tensors, splits and normalization; the fixed network flattens each image in row-major order.

Prediction NPZ has exactly `ids`, `logits`, `seeds`, `bench_seed`, `rates`, and `config`. Write through an open binary file handle to the exact requested path.

- `ids`: exactly the ordered input IDs.
- `logits`: finite float32 `(25,N,10)` final-epoch logits, one independently trained member per slot; no ensembling.
- `bench_seed`: positive integer scalar; `seeds`: ordered integer vector `42+25*(bench_seed-1)+r`, r=0..24.
- `rates`: float64 `(25,6)`, one shared allocation repeated across replicas, each rate in[0,1), row mean0.1.
- `config`: scalar JSON object exactly matching frozen `schema.CONFIG`. This records the fixed6-hidden-layer width256 critical ReLU network, CE objective and matched training controls; it is mandatory.

The75-epoch final endpoint, batch100, AdamW/cosine settings, float32 arithmetic and critical initialization parameters are experimental controls of the dropout-allocation comparison. Step(early) rates are the baseline solution: any other lawful allocation/search method preserving the fixed budget is allowed. Supplied train/validation use and independent seed mapping remain unchanged.

The model directory must contain `model_card.json` with `format="eoc-mlp-v1"`, the same `config`, `bench_seed`, `seeds` and `rates`, plus25 ordered `models`. Each entry supplies its `seed`, `epoch=75`, relative tensor checkpoint `file` and SHA256 digest. Tensor keys are `layers.i.weight`/`layers.i.bias`, i=0..6, with dimensions3072→256→256→256→256→256→256→10. Checkpoints must load with `torch.load(weights_only=True)`.

`check_model.py DIR --input INPUT --pred PRED` validates artifacts and reconstructs the frozen ReLU forward pass without candidate imports, requiring agreement with submitted logits. The generic arbitrary-model certificate is not accepted. Training provenance requires source audit. Final weights alone cannot establish initialization, real use of dropout, optimizer settings, independent training, or lawful data use.

`validate.py PRED INPUT` validates tensor format and alignment. `scorer.py PRED LABELS` additionally checks the allocation/configuration and computes the unchanged mean member cross-entropy score `100*(1-meanCE/2.20679)`. Invalid predictions or contracts fail with Infinity.

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
