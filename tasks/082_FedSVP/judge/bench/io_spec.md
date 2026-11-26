# Frozen I/O contract

Inputs are NPZ archives loaded without pickle: `id` is an int64 vector of unique,
ascending original row IDs; `x` is a finite float32 matrix with 512 columns of
normalized frozen CLIP ViT-B/16 embeddings. Each prediction depends on its own
feature row and the fitted model. Input IDs are identifiers, never features.

Training archive `data/train/train.npz` has `id,x,y,client,rank`. Classes `y` are
integers 0..100; `client` is 0..9; `rank` preserves the original within-client
sample ordering. `data/text.npz` contains normalized `text`, shape (101,512).
Validation labels CSV has exactly `id,y,client`; validation input contains no
labels or client membership. `data/val/refit.npz` supplies the original training
augmentation and memberships for validation diagnostics only. The historical
filename does not authorize using these validation rows for fitting.

Prediction is UTF-8 CSV with header:
`id,global,personal_0,personal_1,personal_2,personal_3,personal_4,personal_5,personal_6,personal_7,personal_8,personal_9`.
All values are integers. Each class prediction is 0..100. Exactly one row per
input ID in ascending order is required. No duplicates, omissions, extra rows,
NaN or infinities are allowed. Write the exact supplied output path, regardless
of its suffix. Predict all ten personal heads; the scorer privately selects the
head belonging to the row's client. Client membership is not prediction input.

GenAcc = 100 times global correct / row count. PerAcc = arithmetic mean of ten
client accuracies, each using its corresponding personal head. Every client must
have rows. PerAcc below 86.5 is a guard failure. The semantic worst score is 0;
the harness failure signal is `{"genacc": Infinity}` to prevent invalid runs
from being averaged as successful submissions. Structural guards need the probe
and patch audit; predictions alone cannot certify training history.

The baseline model directory contains `model.npz` and `model_card.json`.
Certificate format `fedspa-feature-card/v1` declares frozen encoder, 101 classes,
512 dimensions, 8-shot total pool, 10 clients, rho 0.1, ten rounds, BENCH_SEED,
alpha/beta, training IDs and 100 upload shape/count records. The probe rejects training IDs
outside `data/train/train.npz`; pass `--data DATA` or set `BENCH_DATA_DIR`. The NPZ contains
`global,text,adapter_0..9,labels_0..9,semantic_0..9`; each tensor is float32,
(101,512), except labels tensors (101,101). The personal semantic states precede
the final server update. The probe independently recomputes all predicted classes.
For alternative predictors on the same fixed frozen features and federated protocol, expose `code/model_api.py` with `predict(model_dir,x)` accepting NumPy float32 `[N,512]` features and returning `[N,11]` integer class predictions (global then ten personal heads). Fitted state may use any candidate-understood format; the legacy `state: "model.npz"` card field names the state bundle. Retain the same input/federation/communication certificate fields, training IDs and upload accounting; alpha/beta and baseline tensor keys are optional in this generic path. The isolated `check_model.py --code CODE` probe checks full/subset prediction consistency without importing candidate code into the trusted scorer. No prototype, cache or dot-product architecture is required. Actual data use and communications remain subject to patch audit.

## Research scope

Objective: Improve federated UCF101 generalization on frozen CLIP ViT-B/16 features while retaining personalized accuracy.

Fixed research objects and target: The supplied frozen image/text embeddings, eight-shot total labeled pool, ten clients with fixed rho=0.1 assignment, ten communication rounds and per-client-per-round upload limit 101*512 floats remain fixed. Global accuracy is the objective; the unweighted ten-client personalized accuracy must be at least 86.5%.

Editable methods: The learned heads, prototype/adaptation method, training surrogate, local optimizer and aggregation strategy may change within the fixed feature, federation and communication setting.

Forbidden substitutions: Do not change or fine-tune the CLIP encoders, replace the supplied features, centralize private client inputs, change client assignments/rounds/uploads, expose private client membership at prediction, or alter either accuracy rule. When retaining the supplied FedSPA pipeline, changing only alpha=9, beta=8, local_lr=0.0004, global_lr=0.0004, lam_reg=0.2, temperature tau=0.01, local_batch_size=8, global_batch_size=8, local_epochs=5, local_epochs_last=100 or local_epochs_server=100 does not count as a method improvement.

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
