# FedQueue I/O specification

## Data

`data/train/train.npz`: `id` int64[N], `x` uint8[N,28,28], `y` uint8[N].
`data/val/input.npz`: `id` int64[N], `x` uint8[N,28,28].
`data/val/labels.npz`: `id` int64[N], `y` uint8[N], scalar
`baseline_mean_total_local_steps` (the fixed local-work reference).
Training and validation IDs are original source row numbers, strictly increasing.
Convert pixels to float32, add the channel axis, divide by 255. No further
normalization or augmentation is used by the baseline. Targets are classes 0–9.
The judge supplies the same input schema; do not make assumptions about its IDs
or row count. Fit only on the supplied training file. Use validation only for evaluation
and method, hyperparameter or checkpoint selection.

## Predictions

`predict.sh --model DIR --input PATH --output FILE` writes a CSV to the exact
output path, regardless of its suffix, with this exact header:

```csv
id,pred,total_local_steps,rounds,clients,protocol,selected_round
```

One row for every input ID, in increasing ID order; no missing, duplicate or extra
rows. All fields except `protocol` are integers. `pred` is the predicted class in
0–9. The remaining fields are constant across rows:

- `total_local_steps`: sum of scheduler `Ek` records over the entire completed run,
  including returned shutdown updates. Preserve the original scheduler accounting;
  this is distinct from optimizer calls and from steps at the selected round.
- `rounds`: 50 target aggregation rounds.
- `clients`: 4.
- `protocol`: `fedqueue-alpha01-rho09-v1`.
- `selected_round`: 1–50, earliest maximum validation accuracy among target rounds.

Scoring computes `accuracy = 100 * mean(pred == y)`. Each full run must satisfy `total_local_steps <= 1.2 * baseline_mean_total_local_steps`. Protocol violations
produce `accuracy: Infinity` (harness failure sentinel), `manifest_score: 0`, and
`guard_pass: false`. Valid runs have `manifest_score == accuracy`. The harness
averages accuracy over training seeds; it never selects a round from hidden labels.

## Model artifact and probe

The baseline saves `model.ts` (TorchScript, normalized float32[N,1,28,28] input,
[N,10] logits output), all round checkpoints, `model_card.json`,
`round_history.jsonl`, `logs/fedqueue_rounds.csv`, `queue_draws.jsonl`,
`partition_ids.npz`, and `selection.npz` (`id`, `pred` of shape [50,N_val]).

`model_card.json` has format `fedqueue-card/v1`; see `code/main.py` for export.
It records seed, protocol, rounds, clients, alpha1/alpha2, queue-environment constants and the actual scheduler configuration,
full-run steps, selected round, validation curve, partition sizes, and SHA256 hashes
of the model, partition IDs, step log, and queue log. The selected model must reproduce
the prediction file. Queue events list `cid` and `delay` in RNG draw order.

Run `python bench/check_model.py DIR --input data/val/input.npz --pred FILE
--data data --seed S` in the reference-compatible environment. It independently
recounts scheduler steps, replays the partition and queue RNG, checks validation
selection, and recomputes predictions from saved fitted state. Current harness versions do
not automatically invoke this probe: these constraints also require patch audit.
A self-reported log or certificate alone cannot establish that training executed it.
The probe loads the saved state dictionary from `model.ts` into the frozen Table-9 SimpleCNN and recomputes predictions with that architecture. A different TorchScript forward or `code/model_api.py` cannot replace the research model. Keep cross-entropy as the learning objective and minibatch-64 as the local-step accounting convention; these training semantics require source review. Optimizer, aggregation and queue-aware scheduling may vary while the fixed partition, exogenous queue environment, genuine compute-time feedback, 50 target rounds and total-work limit remain unchanged.

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
