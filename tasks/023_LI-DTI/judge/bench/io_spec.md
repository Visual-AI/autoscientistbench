# LI-DTI numerical prediction interface

`train.sh --out DIR` saves method artifacts for the fixed 10 repetitions ×10 folds; the baseline writes `DIR/model.mat`. The explicit harness smoke flag selects the small one-fold real-data fixture. `predict.sh --model DIR --input FILE --output FILE` computes fresh predictions with the submitted method, which may replace the original fold-weight model.

Input is CSV with exact header `run,fold,sample,row,col`. All coordinates are positive one-based integers in the already filtered dataset. Output is CSV with exact header `run,fold,sample,row,col,pred`, one finite numeric score for every input row, no extras or duplicates. The sample ID preserves repeated occurrences in the authors’ splits; pair coordinates may repeat, while the full key must be unique. Row order is arbitrary. Scores are real-valued rankings and need not lie in [0,1]. Labels use the same coordinates and final column `y` (0/1).

Full inputs contain100 original fold intersections with a permanent outer pair partition. Every negative in each intersection is included. Candidate training data contains only globally masked interactions and sorted two-column author fold masks. The original label column and positive-first ordering are removed. All hidden-positive entries are zero in every public training artifact.

The scorer groups by run and fold, computes one point at the last item of each tied score group, integrates precision over recall using trapezoids, and does not prepend (0,1). It averages10 fold scores in each run, then10 run means. `AUPR_std` is the sample standard deviation across run means. Exact key matching rejects missing, extra and duplicate predictions; nonfinite values are invalid. Invalid outputs score0 and fail validation.

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
