# I/O specification

`data/train/graph.npz` contains `ids` (opaque strings in graph row order),
`features` (float32 node-by-feature matrix) and `adjacency` (float32 symmetric
matrix with self loops). There are no truth labels in this file. The baseline
uses the single graph twice, as in the released loader.

`--input` is a CSV with header `id`, listing requested nodes. Both development
and final evaluation request the entire graph. The scoring mask is held only
in the evaluator's labels file. Predict from the model directory; never inspect
labels during prediction.

`--output` is CSV with exactly `id,pred_0,pred_1,pred_2,pred_3,pred_4`.
Each column is an independent clustering run, each entry an integer in 0..4.
Return exactly the input IDs, in input order, without duplicates or missing
values. No NaN or Infinity. Every full-graph column must use all five clusters.
Write the exact requested output filename, even if it has no extension.

The frozen scorer takes `scorer.py PRED LABELS`, where labels are CSV `id,y`
and can cover a subset of the prediction IDs. It separately Hungarian-matches
each column using only labeled rows, then computes ACC and macro-F1 with
sklearn. The metric `acc` is mean ACC across the five columns if mean macro-F1
is at least 0.30 and each full-graph clustering uses five clusters. A violated
guard produces Infinity (invalid submission), plus diagnostics including raw
ACC, macro-F1 and cluster counts. The judge averages over three outer seeds.

Fitting and checkpoint selection must remain unsupervised. Public validation
labels support evaluation and method or hyperparameter selection only.
Training must save all fitted state under `--out`. The baseline saves five
checkpoints with the selected assignments, model/optimizer state, noisy inputs,
embeddings, RNG state and unlabeled selection score. Returning saved full-graph
assignments learned in this run is valid for this transductive task.

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
