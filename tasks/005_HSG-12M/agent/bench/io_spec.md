# HSG classification I/O

Each record is one spatial multigraph. `input.npz` is a non-pickled NumPy archive with opaque string `id[N]`, packed tensor arrays `data_NAME`, and int64 offsets `ptr_NAME[N+1]`. Graph i occupies the half-open slice between adjacent offsets. Node indices in `data_edge_index` are local to each graph; add the cumulative node offset when constructing a batch.

The node features `data_x` have four float32 columns: two spatial coordinates, potential and density of states. `data_edge_attr` has six float32 columns: weight, displacement, midpoint coordinates, average potential and average density of states. The remaining supplied tensor fields retain the authors' preprocessing. `data_edge_index` is an int64 2-by-E directed edge list; `data_pos` holds node positions, and `data_n_nodes` / `data_n_edges` give graph sizes. The input has no targets or source dataset indices. The graph order is the authors' fixed batch rebalancing order.

`labels.csv` has header `id,y`, with integer classes 0 through 23. Training and validation both use `input.npz` and `labels.csv` under their respective directories.

Prediction is UTF-8 CSV, one row per supplied id, with this exact header:

```text
id,pred,parameters,steps,model_sha256,epochs
```

`pred` is an integer class in 0..23. `parameters` counts all trainable parameters used in the fit, `steps` counts updates to the fitted model across all ranks, with a synchronized DDP step counted once, and `epochs` counts epochs started, including a final partial epoch. These three fields are nonnegative integers. `model_sha256` is the lowercase SHA-256 of the fitted-state artifact identified by the model card. All metadata must be identical across rows. Every input id must occur exactly once, in any order. Values must be finite. Output paths may have any extension or no extension.

`train.sh --out DIR` exports all fitted state and a `model_card.json`. Any model, surrogate loss, optimizer or training algorithm may be used within the fixed capacity and training-work controls. The certificate has `format: "method-model-v1"`, `artifact_file` naming one fitted-state bundle inside the model directory, `weights_sha256` equal to that file's SHA-256, and nonnegative integer diagnostics `trainable_parameters`, `optimizer_steps`, `epochs_started`. The counts are binding: at most 731,203 real trainable parameters, 1,000 updates across all ranks, with a synchronized DDP step counted once, and 100 started epochs. Zero is valid only when that work or fitted parameter genuinely does not exist. The predictor's metadata must match this card. The bundle may use any representation understood by candidate code; the trusted scorer/probe only hashes it, without importing or deserializing candidate objects. Generic certificates also name `parameter_artifact_file` (default `trainable_parameters.npz`): a non-pickled NPZ containing every learned scalar used in fitting, including learned constants/buffers. The probe independently counts numeric entries (complex entries count twice) and compares the count with the card. Parameter completeness and execution-work provenance require source audit.

The baseline's `hsg-model-v1` certificate remains supported: `weights.pt` is a tensor state dictionary; `parameters` and `buffers` map all tensor names to shapes. The probe validates this manifest and count when that legacy format is supplied. Distributed-rank traces are baseline diagnostics rather than required method structure.

The scorer returns the arithmetic fraction of correct class predictions. Aggregation across fits is the arithmetic mean. The scorer rejects parameter, update or epoch counts beyond the fixed controls. The original supplied graph inputs, training/validation label-use rules, 24 classes and accuracy formula remain fixed. Model artifact provenance requires patch review.

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
