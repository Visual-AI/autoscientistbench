# Aitchison Embeddings for Learning Compositional Graph Representations

Task statement for the agent. Everything below is binding: the judge reruns your `code/` from the prescribed
starting state with its own seeds, scores the hidden pairs, and compares with the archived baseline.

## 1. Goal

How well can an 8-dimensional node representation, learned from the residual graph alone with no node features,
predict edges removed from that graph? The graph is Cora.

You output a frozen representation for every node: 8 numbers, one scalar bias, and the name of a decoder from a small
frozen family. You never see a list of pairs to score. The scorer holds the pairs, computes their scores from your
file, and reports AUC-ROC. Improve `auc_roc` (higher is better, bound 1.0). The supplied baseline is the paper's
AICoG (LB) at K = 9; you may replace it entirely within the research contract below.

## 2. Metric and guards

`auc_roc`: AUC-ROC over the scorer's pairs, with tied scores counted one half. Pair scores are computed in float64 from
your `z`, `b` and decoder. See `bench/io_spec.md`.

| Guard | On violation | Enforced by |
|---|---|---|
| `z` has shape (N, 8) and `b` has shape (N,): 8 dimensions per node, one scalar per node | Infinity / invalid | scorer and `validate.py` |
| `z` and `b` are finite floats with magnitude at most 1e100 | Infinity / invalid | scorer and `validate.py` |
| `decoder` is one of the two frozen names | Infinity / invalid | scorer and `validate.py` |
| The file holds exactly `z`, `b`, `decoder`, with no pickled objects | Infinity / invalid | scorer and `validate.py` |
| The metric is computed by the scorer from pairs it holds; nothing claimed in your file is read | not possible | scorer |
| `predict.sh` receives only the node count, never a pair list | not possible | interface |
| Models and learned preprocessing are fitted only on `data/train/`. Val positives are true edges and must not be added to the training graph; no final train+val refit | disqualified | patch audit |
| No node features, node labels or external copy of Cora, including matching the relabelled graph against a public copy | disqualified | patch audit |

The judge scores 5 seeds on the hidden pairs and compares the mean with the archived baseline. PASS means a relative
improvement of at least `max(delta, 2 × reference σ / reference mean)`, with delta = 0.011. `bench status` shows both.

## Research contract

### Objective

Improve link-prediction AUC-ROC on the supplied Cora graph using an 8-dimensional node representation learned from
training edges alone, without node features. AICoG is the supplied baseline; other representation methods are allowed.

### Fixed research objects

These define the problem; a patch that changes them is invalid:

- the representation budget: 8 dimensions per node, plus at most one scalar per node, the bias term the baseline uses;
- the input of learning: only the training edges and the node count, with no node features, no labels and no external
  copy of Cora;
- the decoder family, the metric and the output format;
- `data/`, `bench/`, this README and the script signatures.

### Editable methods

Under `code/`, everything about how the representation is learned may change:

- objective and loss, geometry, optimizer and schedule, initialization;
- graph preprocessing derived from the training edges;
- model selection on `data/val/`: hyperparameters of a changed method, checkpoints and the choice of decoder;
- ensembling, provided the output still has 8 dimensions per node;
- dependencies through `env.lock`, and any numerical implementation.

### Forbidden substitutions

Changing the fixed research objects; fitting on validation data, including adding val positives to the training
graph, a final train+val refit, or distillation from validation; obtaining or embedding the full graph by any route;
hard-coding pairs, node scores or results. When retaining the supplied AICoG training procedure, changing only lr=0.01, epochs=5000, scaling_epochs=500 or M_neg=5*sparse_i_idx.shape[0] does not count as a method improvement.

## 5. Data

- `data/train/edges.csv`: the authors' residual graph, unchanged: 2535 undirected edges on 2485 nodes, connected.
- `data/train/meta.json`: the node count.
- `data/val/input.json`, `data/val/labels.npz`: 253 removed edges and 253 sampled non-edges, with labels. Use them to
  evaluate and select, not to fit.
- The hidden pairs are held by the judge and are never readable from this workspace.

All node ids come from a secret random relabelling, so they do not match public copies of Cora.

## 6. Baseline

- Method: AICoG (LB), K = 9 (D = 8), the only variant the released code trains: per-node logits, softmax composition, ILR
  coordinates with a learned orthonormal basis, node biases, and a Bernoulli likelihood whose log-partition term uses
  uniformly sampled distinct-node pairs.
  5000 epochs, the first 500 fitting only the biases, Adam at lr 0.01; these are the defaults in `main.py`.
- Official code: https://github.com/Nicknakis/AICoG at commit `890a99e5a810b7ab2cc6306ff9c41536e9fdf748`, under `code/`.
  The model is `src/AICoG.py`; the training loop is `main.py`.
- It ranks pairs by `neg_euclidean`: the score its own `link_prediction()` computes, from `z` = ILR coordinates and `b` = biases.
- `main.py` reads the training edge list and seed from its arguments and writes `embedding.npz`.
  `predict.sh` copies that frozen representation to the output format in `bench/io_spec.md`.
- Measured reference on validation, seeds 1 to 5: 0.8064, 0.8423, 0.8152, 0.8264, 0.8144.
  Mean AUC-ROC is 0.820938, with standard deviation 0.012430. `bench status` shows the reference.
- With 4 CPUs, training times for those seeds are 66.86, 64.26, 64.81, 69.47, 64.51 s.
  Validation prediction times are 0.56, 0.56, 0.52, 0.56, 0.51 s.

Val contains 253 positive and 253 negative pairs. Compare methods on the same val pairs;
a small validation gain may not generalize.

## 7. Interface

- `bash code/train.sh --out DIR` writes everything `predict.sh` needs into DIR. The baseline writes `embedding.npz` and
  a checkpoint.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes the npz in `bench/io_spec.md`. The input holds
  only the node count.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR` (`data/`), `BENCH_BENCH_DIR` (`bench/`), `BENCH_CODE_DIR` (`code/`).
- `bench status` · `bench eval-dev [--seed S]` (train, predict, validate and score on val) · `bench validate-output FILE`
  · `bench submit [--final] [-m MSG]`.
- Dependencies: `uv pip install ...` into the venv; `bench submit` freezes them into `code/env.lock`. Only `name==version`
  pins from PyPI or the PyTorch CPU index are rebuildable.

## 8. Resources

4 CPUs and 4 GB of memory. No network during train or predict. There is no separate stage limit: use the overall deadline
as you see fit. Evaluation has its own overall budget covering all 5 seeds, so a very slow method can exhaust it. Submit a
valid patch with `bench submit` before the deadline; no valid submission means failure.

## 9. Seeds

The judge runs five seeds, and every run must honor `BENCH_SEED`. Variance across seeds counts against you.

The baseline is not seed-deterministic: the same seed gives different embeddings. `scipy.sparse.linalg.eigs` draws its
start vector from OS entropy, and seeding it alone did not make multi-threaded runs identical. Runs of seed 1 gave 0.8330,
0.8349, 0.8104 and 0.8094 on validation. The judge's check that the baseline still reproduces therefore allows an absolute
0.015 on top of σ/2.

## 10. Notes

- The training graph is close to a tree: about half the nodes have degree 1. Held-out edges close cycles in it.
- Every method must use one of the frozen decoders in `bench/io_spec.md`; `inner_product` is the alternative to distance.
  This output format cannot express a logistic regression on pair features.
- `main.py` keeps the authors' node-classification branch. It needs labels that are not shipped, and labels are not
  allowed as input here.
- The authors' README describes dataset folders that are not shipped. Read `data/train/edges.csv` instead.
- `PYTHONSAFEPATH=1` is set, so the entry scripts export `PYTHONPATH` themselves. Only committed files reach the judge's
  clean checkout, and `.gitignore` excludes `*.pt`.
- Write the output through an open handle (`bench/io_spec.md`); `np.savez` appends `.npz` to a bare path.

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
