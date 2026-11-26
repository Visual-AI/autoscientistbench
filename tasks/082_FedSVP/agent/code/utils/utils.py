from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# CPU/GPU-agnostic device selection (the original code hard-codes .cuda()).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_shot_accuracy(image_embeds: torch.Tensor,
                       prototype_embeds: torch.Tensor,
                       gts: torch.Tensor) -> float:
    """
    Zero-shot prediction via cosine similarity between image and prototype embeddings.

    Args:
        image_embeds: Tensor (N, D)
        prototype_embeds: Tensor (C, D)
        gts: Tensor (N,) or list of length N

    Returns:
        accuracy: float, percentage in [0, 100]
    """
    img_norm = F.normalize(image_embeds, p=2, dim=1)
    proto_norm = F.normalize(prototype_embeds, p=2, dim=1)
    cos_mat = img_norm @ proto_norm.T
    preds = cos_mat.argmax(dim=1)
    correct = (preds == gts).sum()
    acc = correct.float() / preds.size(0) * 100
    return acc


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def cls_acc_2(output, target, topk=1):
    """Compute top-k classification accuracy; returns (correct, total)."""
    pred = output.topk(topk, 1, True, True)[1].t()
    target_ = target.view(1, -1).expand_as(pred)
    correct = pred.eq(target_)
    correct_count = correct[:topk].reshape(-1).float().sum(0, keepdim=True).item()
    total = target.size(0)
    return int(correct_count), total


def distribution_label_skew_split_consistency(
    labels: torch.Tensor,
    num_clients: int,
    alpha: float,
    consistency_seed: int = None,
) -> List[torch.Tensor]:
    """
    Distribution-based label-skew partition across clients.
    consistency_seed aligns per-class allocation ratios between train and test splits.
    """
    labels_np = labels.cpu().numpy()
    classes = np.unique(labels_np)

    client_indices = [[] for _ in range(num_clients)]

    for m in classes:
        idx_m = np.where(labels_np == m)[0]
        np.random.shuffle(idx_m)
        n_m = len(idx_m)

        if consistency_seed is not None:
            rng = np.random.default_rng(seed=consistency_seed + int(m))
            p = rng.dirichlet([alpha] * num_clients)
        else:
            p = np.random.dirichlet([alpha] * num_clients)

        alloc = np.floor(p * n_m).astype(int)

        allocated = alloc.sum()
        if allocated < n_m:
            leftover = n_m - allocated
            if consistency_seed is not None:
                leftover_clients = rng.choice(num_clients, leftover, replace=True)
            else:
                leftover_clients = np.random.choice(num_clients, leftover, replace=True)
            for i in leftover_clients:
                alloc[i] += 1

        start = 0
        for k in range(num_clients):
            cnt = alloc[k]
            if cnt > 0:
                client_indices[k].extend(idx_m[start : start + cnt])
                start += cnt

    return [torch.tensor(ci, dtype=torch.long) for ci in client_indices]
