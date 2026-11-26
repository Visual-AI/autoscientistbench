# TT-Sparse I/O specification

Inputs are CSV with header `id,preg,plas,pres,skin,insu,mass,pedi,age`. Each row has an opaque unique id and eight raw Pima numeric features. No target is supplied during prediction. Input size and order may vary.

Training uses `data/train/train.csv` (same columns plus `target`, with string classes `tested_negative` and `tested_positive`). Validation has `input.csv` and `labels.csv`. Labels have `id,y` followed by the eight raw features; y=1 means tested_positive. The duplicated features let the frozen scorer recompute rule predictions without loading candidate code. `data/train/development_order.json` lists the supplied training and validation ids in preprocessing order. It is not a source row index.

Prediction is CSV text, even if the requested output filename has no extension. Exact header:

```
id,certificate,pred_0,pred_1,pred_2,pred_3,pred_4
```

Each input id appears exactly once, in any order. The five `pred` columns are the positive-class probabilities for five independently trained replicates; never average probabilities across replicates. Every probability is finite and in [0,1]. Exactly one row has a nonempty `certificate` cell containing a JSON object, CSV quoted; other certificate cells are empty. Use `certificate.write_prediction` for this serialization.

Certificate schema (also saved as `rules.json` in the training output directory):

- `format`: `tt-sparse-rules-v1`.
- `models`: exactly five entries ordered by replicate 0 through 4.
- Each entry has `encoder` and `rules`. `encoder` contains ordered `features`, eight `mean` and positive `scale` values, `num_bits`, eight arrays of `thresholds` in standardized units, and `classes` in negative/positive order. Numeric encoder semantics are frozen in `frozen_encoder.py`; they preserve the authors' float32 skip features and exact thermometer comparisons.
- `rules` is a learned rule list, obtained by any method (the baseline uses `explain`): each entry has `kind` (`intercept`, `skip`, or `ltt`), `r` (expression), and `w` (weight dictionary keyed by `tested_positive`). There is one `True` intercept. Skip expressions name raw features; LTT expressions use the authors' Boolean grammar, including XOR simplification.

The frozen scorer computes ROC-AUC separately for each replicate, then the arithmetic mean. Complexity is recomputed from actual expressions: 1 per intercept/skip, and `1 + count('&') + count('|') + count('^')` per LTT rule, exactly as the authors do. The guard applies to the five-replicate mean, not each individual replicate. Predictions are recomputed through the frozen upstream `predict_rules`; maximum absolute disagreement must not exceed 1e-12. A guard violation produces `auc: Infinity` (harness failure sentinel), with `paper_score: 0.5` preserving the paper protocol's worst score as a diagnostic.

`check_model.py MODEL --input INPUT --pred PRED` binds the prediction certificate to the training artifact. Training provenance and honoring independent training seeds are checked by patch audit. Direct sparse-rule learning or distillation from any model is allowed, provided the resulting certificate has the same rule semantics and complexity accounting. A certificate must be learned during training and remain identical across prediction inputs. No per-input fitted certificate, memorized predictions, external data, or hidden-target reconstruction is permitted.

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
