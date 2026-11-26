"""Frozen authors' MCC implementation; numerical conventions intentionally preserved."""
import numpy as np
import torch

def _standardize(x):
    """Standardize columns. x must be torch.Tensor (use _to_tensor upstream if needed)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)


def pearson_corr(x, y):
    """
    x, y: (T, d) torch.Tensor or np.ndarray (converted to torch).
    returns: (d, d)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(np.asarray(y, dtype=np.float64))
    x = _standardize(x)
    y = _standardize(y)
    return x.T @ y / x.size(0)


def rankdata(x):
    """
    Column-wise ordinal ranking. x must be torch.Tensor (use conversion upstream if needed).
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    return torch.argsort(torch.argsort(x, dim=0), dim=0).float() + 1.0


def spearman_corr(x, y):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(np.asarray(y, dtype=np.float64))
    return pearson_corr(rankdata(x), rankdata(y))


# ============================================================
# Auction assignment (permutation matching)
# ============================================================

def auction_assignment(x, eps=None, max_iter: int = 10000):
    """
    Solve max-sum assignment using auction algorithm.
    x: (d, d) torch.Tensor (or converted from np.ndarray).
    max_iter: hard cap to avoid infinite loop on ties/near-ties.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    d = x.size(0)
    eps = 1.0 / d if eps is None else eps

    price = torch.zeros((1, d), device=x.device)
    assignment = torch.full((d,), -1, device=x.device, dtype=torch.long)
    bids = torch.zeros_like(x)
    it = 0
    while (assignment == -1).any() and it < max_iter:
        it += 1
        unassigned = (assignment == -1).nonzero(as_tuple=False).squeeze(1)
        if unassigned.dim() == 0:
            break

        value = x[unassigned] - price
        top2_val, top2_idx = value.topk(2, dim=1)

        best_item = top2_idx[:, 0]
        bid_inc = top2_val[:, 0] - top2_val[:, 1] + eps

        bids.zero_()
        bids[unassigned, best_item] = bid_inc

        items = (bids > 0).any(dim=0).nonzero(as_tuple=False).squeeze(1)
        if items.numel() == 0:
            break
        max_bid, winner_idx = bids[:, items].max(dim=0)

        price[:, items] += max_bid

        assignment[torch.isin(assignment, items)] = -1
        assignment[winner_idx] = items

    return assignment


# ============================================================
# Permutation-invariant latent recovery metric (MCC)
# ============================================================

def mean_corr_coef(z_true, z_hat, method="pearson"):
    """
    MCC: Mean Correlation Coefficient. Valid only when z_true has identifiable
    ground-truth semantics and z_hat is recovering those factors (e.g. synthetic
    factor recovery, oracle experiments). Do not use for cross-model stability
    or representation-similarity comparison.
    z_true, z_hat: (T, d) torch.Tensor or np.ndarray (converted to torch).
    """
    if isinstance(z_true, np.ndarray):
        z_true = torch.from_numpy(np.asarray(z_true, dtype=np.float64))
    if isinstance(z_hat, np.ndarray):
        z_hat = torch.from_numpy(np.asarray(z_hat, dtype=np.float64))
    if method == "pearson":
        C = pearson_corr(z_true, z_hat)
    elif method == "spearman":
        C = spearman_corr(z_true, z_hat)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    C = C.abs()
    assignment = auction_assignment(C)

    return C[torch.arange(C.size(0)), assignment].mean()


