#!/usr/bin/env python3
"""Frozen scorer: AUC-ROC of a frozen node representation on pairs held by the scorer.

    python scorer.py PRED LABELS   -> one JSON line {"auc_roc": value, ...}

The prediction holds z (N x 8), b (N) and the name of one decoder. Pair scores are computed here, and AUC-ROC by the
tie-correct rank formula below. Nothing else in the file is read, and no metric is taken from it. Any violation prints
auc_roc = Infinity, which the harness treats as a failed submission.
"""
import json
import sys
import zipfile

import numpy as np

METRIC = "auc_roc"
EMBED_DIM = 8
MAX_ABS = 1e100     # with this cap no pair can overflow under either decoder, so validate.py and the scorer always agree
MEMBERS = {"z.npy", "b.npy", "decoder.npy"}


def neg_euclidean(z, b, i, j):
    """b_i + b_j - ||z_i - z_j||_2. AICoG's log-odds in ILR coordinates, where Aitchison distance is Euclidean."""
    return b[i] + b[j] - np.sqrt(((z[i] - z[j]) ** 2).sum(axis=1))


def inner_product(z, b, i, j):
    """b_i + b_j + z_i . z_j"""
    return b[i] + b[j] + (z[i] * z[j]).sum(axis=1)


DECODERS = {"neg_euclidean": neg_euclidean, "inner_product": inner_product}


def load_prediction(path, n_nodes):
    """-> (z float64, b float64, decoder name). Raises ValueError on anything outside the format."""
    with open(path, "rb") as f:
        if f.read(4) != b"PK\x03\x04":
            raise ValueError("not an npz archive")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not an npz archive: {exc}")
    if len(names) != len(set(names)) or set(names) != MEMBERS:
        raise ValueError(f"members must be exactly {sorted(MEMBERS)}, found {sorted(names)}")
    with np.load(path, allow_pickle=False) as d:
        z, b, decoder = d["z"], d["b"], d["decoder"]
    if z.dtype not in (np.float32, np.float64) or b.dtype not in (np.float32, np.float64):
        raise ValueError(f"z and b must be float32 or float64, found {z.dtype} and {b.dtype}")
    if z.shape != (n_nodes, EMBED_DIM):
        raise ValueError(f"z must have shape ({n_nodes}, {EMBED_DIM}), found {z.shape}")
    if b.shape != (n_nodes,):
        raise ValueError(f"b must have shape ({n_nodes},), found {b.shape}")
    if not (np.isfinite(z).all() and np.isfinite(b).all()):
        raise ValueError("z and b must be finite")
    if max(np.abs(z).max(), np.abs(b).max()) > MAX_ABS:
        raise ValueError(f"values of z and b must not exceed {MAX_ABS:g} in magnitude")
    if decoder.dtype.kind != "U" or decoder.shape != ():
        raise ValueError(f"decoder must be a 0-d unicode array, found dtype {decoder.dtype} shape {decoder.shape}")
    name = str(decoder)
    if name not in DECODERS:
        raise ValueError(f"unknown decoder {name!r}; allowed: {sorted(DECODERS)}")
    return z.astype(np.float64), b.astype(np.float64), name


def auc_roc(scores, y):
    """Mann-Whitney AUC with average ranks for ties."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_rank = (ends - counts + ends + 1) / 2.0
    ranks = average_rank[inverse]
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score(pred_path, labels_path):
    with np.load(labels_path, allow_pickle=False) as d:
        i, j, y, n_nodes = d["i"], d["j"], d["y"], int(d["n_nodes"])
    z, b, name = load_prediction(pred_path, n_nodes)
    scores = DECODERS[name](z, b, i, j)
    if not np.isfinite(scores).all():
        raise ValueError("decoder produced non-finite pair scores")
    return {METRIC: auc_roc(scores, y), "decoder": name}


def main(pred_path, labels_path):
    try:
        report = score(pred_path, labels_path)
    except Exception as exc:  # noqa: BLE001 -- every failure is a guard violation and must not look like a crash
        report = {METRIC: float("inf"), "reason": f"{type(exc).__name__}: {exc}"[:500]}
    print(json.dumps(report))


if __name__ == "__main__":
    main(*sys.argv[1:3])
