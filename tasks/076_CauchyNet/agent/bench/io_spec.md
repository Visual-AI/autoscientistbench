# Input and output

Input is CSV with exactly `id,x`. Each row is one scalar coordinate, in natural units.
Training uses `data/train/train.csv` (`id,x,y`). Validation uses `data/val/input.csv`
and `data/val/labels.csv` (`id,y`). Values round-trip to the original float32 data.

Prediction CSV has exactly these columns in order:
`id,pred_0,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,params_0,params_1,params_2,params_3,params_4,params_5,params_6,params_7,params_8,params_9`.
There is exactly one row per input, in the same order. All numbers must be finite.
`pred_s` is the real scalar prediction from independently trained model seed s.
`params_s` is its nonnegative integer real trainable parameter count, constant across rows.
Complex parameters count twice; fixed buffers do not count. Never average model predictions
before writing: the metric averages errors of the ten models, not errors of an ensemble.

Scorer computes each seed's MAE in float32, then averages the ten results in float64.
Each model is limited to 128 real learned parameters; the scorer rejects larger declarations.
The structural probe independently checks actual parameters and prediction consistency.

`train.sh --out DIR` saves all ten independently fitted model states. Any model, surrogate loss, optimizer or method may be used within this same capacity budget. For a general implementation, expose `code/model_api.py` with `predict(model_dir, x)` returning a finite float32-compatible array `[N,10]` for NumPy input `[N,1]`. It must load fitted state from model_dir and use only each row's x; no labels, IDs or other evaluation rows are supplied. Export `model0.parameters.npz` through `model9.parameters.npz` as non-pickled numeric archives of all learned scalar arrays. The probe independently counts their entries (complex entries count twice); each count must agree with the prediction and be at most 128. Optional `parameter_counts(model_dir)` must agree with this count. The isolated candidate probe imports this API and checks full/reversed-subset consistency. The trusted scorer never imports or deserializes candidate state.

The baseline's legacy interface remains supported: save `model0.ts` through `model9.ts` for `torch.jit.load(..., map_location='cpu')`, each accepting float32 `[N,1]` and returning float32 `[N,1]` (or a tuple with it first). The probe validates tensor parameter counts and predictions. Both interfaces use `check_model.py DIR --input PATH --pred FILE --code CODE`. The 128-real-parameter budget applies to both interfaces, including learned quantities later marked frozen or encoded as constants. Manifest completeness and artifact provenance require patch review.

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
