# I/O specification

Inputs are NPZ tensors (`allow_pickle=False`). There are eighteen groups, with
keys `dD_rR_id` and `dD_rR_R`, where D ranges from 0 to 5 and R from 0 to 2.
Dataset order: DHA, AT-AT, stachyose, AT-AT-CG-CG, buckyball-catcher,
double-walled-nanotube; atom counts: 56, 60, 87, 118, 148, 370.
An ID is the string `D:R:original_frame_number`. The coordinate array has shape
(number of IDs, atom count, 3), float32, in Angstrom. Each group is sorted by
original frame number. Predict each group with its corresponding fitted model.

Training files `data/train/dD.npz` contain sorted original integer `id`, raw
float32 `R` coordinates, `E` molecular energy in kcal/mol, `F` force in
kcal/mol/Angstrom, and scalar `n_atoms`. All three replicas may use the same
public training set. Public validation labels are CSV with the header
`id,y,dataset,replica,n_atoms`; y is raw molecular energy, not energy per atom.

Output is text CSV at the exact `--output` path, with header
`id,pred,train_s,peak_gb,cpus`. IDs must match the input exactly, once each;
row order does not matter. `pred` is raw molecular energy in kcal/mol. All
numbers must be finite. Resource fields must be identical on every row.
`train_s` is elapsed wall seconds for the entire eighteen-fit training command;
`peak_gb` is the maximum process-tree memory over training and prediction,
in decimal GB (1e9 bytes); `cpus` is the allocated CPU budget, between 1 and 64.

The frozen `resources.py monitor --report FILE -- COMMAND` produces the resource
record. Its conservative memory estimate sums the high-water RSS of live
processes, sampled every 0.1 s; the child high-water RSS covers short-lived
children. The launcher must wrap all work in this monitor. After monitored
prediction, `resources.py format --raw FILE --output FILE --training FILE
--prediction FILE` attaches the records. Train writes `resources.json` into the
model directory. Predictor artifacts must contain no evaluation observations.
Resource attestation also requires independent judge measurement or patch audit;
self-reported numbers alone do not prove compliance.

Scoring computes RMSE within each of the eighteen groups, divides by its atom
count, averages the three replica values for each dataset, then sums the six
dataset means. It is not RMSE of an ensemble and is not pooled over datasets.
Memory above 56 GB or a CPU allocation outside 1–64 gives `Infinity`.

Research scope: The fixed target is the original observed molecular energy in kcal/mol, not approximation of a specified GP posterior. The predictor must remain a Gaussian process conditioned on observed energies and forces. Its kernels, derived descriptors, conditioning structure and training surrogate losses may change within the same six-dataset, eighteen-fit and resource-constrained protocol. All fitting remains in train. Original frame numbers in IDs and `id` arrays are row keys only: do not use them as model inputs, to choose conditioning sets, or to infer which frames are held out. The GP prior mean must be fitted on the training split only and must not carry the prediction: no mean or energy offset may be selected on validation, and no neural network, tree ensemble or other non-GP regressor may serve as the mean.

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
