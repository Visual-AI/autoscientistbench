# AICoG task: I/O specification

## Input of `predict.sh`

`--input` is a JSON file holding only the node count:

```json
{"n_nodes": 2485}
```

The validation and hidden inputs are byte-identical. `predict.sh` never receives a pair list, not even in a sealed step.
A mixed list of positives and negatives would leak signal through how often each node appears.

## Output of `predict.sh`

`--output` is an `.npz` archive with exactly these three members:

| Member | Type | Shape | Meaning |
|---|---|---|---|
| `z` | float32 or float64 | (N, 8) | the node representation |
| `b` | float32 or float64 | (N,) | one scalar per node; may be all zeros |
| `decoder` | numpy unicode string | () | one name from the decoder family below |

The scorer loads the file with `allow_pickle=False`. A file is invalid if it has any other member, an object array, a
wrong shape or dtype, a non-finite value, a magnitude above 1e100, or an unknown decoder name.

The output path may have no extension, and `np.savez` appends `.npz` to a bare path. Write through an open handle:

```python
with open(output_path, "wb") as f:
    np.savez(f, z=z, b=b, decoder=np.array("neg_euclidean"))
```

## Decoder family

| Name | Score of pair (i, j) |
|---|---|
| `neg_euclidean` | `b[i] + b[j] - sqrt(sum((z[i] - z[j])**2))` |
| `inner_product` | `b[i] + b[j] + sum(z[i] * z[j])` |

`neg_euclidean` ranks pairs as the authors' `LSM.link_prediction()` does: with `z` in ILR coordinates, Aitchison distance
is Euclidean. `inner_product` lets methods in another geometry compete. The family is frozen.

## Metric

The scorer holds the pairs, computes each pair's score in float64 from `z`, `b` and the named decoder, and returns AUC-ROC
by a tie-correct rank formula (tied scores count one half). No metric is read from the candidate.

```
python bench/validate.py PRED INPUT     # exit 0 when the format is valid
python bench/scorer.py PRED LABELS      # prints {"auc_roc": value, "decoder": name}
```

A violation prints `"auc_roc": Infinity`, which the harness treats as a failed submission.

## Data files

- `data/train/edges.csv`: header `i,j`, then one training edge per row with `i < j`. Node ids are integers from 0 to N-1.
- `data/train/meta.json`: `{"n_nodes": N}`.
- `data/val/input.json`: the input above.
- `data/val/labels.npz`: integer arrays `i`, `j` (`i < j`), `y` (1 = removed edge, 0 = non-edge), and the scalar `n_nodes`.

The hidden labels have the same format and are held by the judge.

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
