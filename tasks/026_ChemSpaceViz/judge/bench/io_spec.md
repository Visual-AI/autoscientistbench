# Fixed-instance PAC layout interface

`--input PATH` is an NPZ file with `allow_pickle=False`:

- `ids`: unique Unicode molecule identifiers, canonical row order.
- `packed`: uint8 array `(n, 128)`, ECFP4 binary fingerprints, 1024 bits, unpacked with `np.unpackbits(..., axis=-1, count=1024)` (big bit order).
- `offsets`: integer cluster boundaries, including 0 and n. Every cluster has more than 20 members. Fixed membership and within-cluster canonical order are part of the instance.

`data/train/input.npz` and `data/val/input.npz` provide the same public fixed instance. `data/val/labels.npz` contains the same input-derived information for the frozen scorer; there are no supervised labels. No coordinates are supplied.

`--output FILE` is UTF-8 CSV with the exact header `id,x,y`. Supply one row per input ID, exactly two finite real coordinates, and no duplicate, missing, or additional IDs. Row order may vary: the scorer restores canonical order before stable ranking. The file name need not have an extension.

`train.sh --out DIR` must write everything needed by `predict.sh` to DIR. The baseline saves a freshly fitted `layout.npz` and seed metadata. `predict.sh --model DIR --input PATH --output FILE` loads these fitted coordinates and aligns the requested IDs.

The scorer computes float32 Tanimoto distances on binary fingerprints, squared float64 Euclidean distances on each cluster's coordinates, and stable double-argsort ranks (ties follow canonical input order). It forms the co-ranking matrix Q including rank zero, removes row and column zero, and computes QNN(k) = sum(Q[1:k+1,1:k+1]) / (n*k), k=1,...,n-1. A cluster's AUC is the arithmetic mean of QNN; `auc_mean` is the unweighted mean over fixed clusters. `auc_size_weighted` is the secondary tie-break value. Rank-zero treatment for duplicate fingerprints follows the original implementation unchanged.

Invalid outputs have literal raw `score=0`, `valid=false`, and `auc_mean=Infinity` as the harness rejection sentinel. Valid outputs have `score=auc_mean`. The validator rejects malformed predictions before scoring.

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
