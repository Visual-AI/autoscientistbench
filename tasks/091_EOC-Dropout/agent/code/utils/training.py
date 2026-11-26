"""Batch training, evaluation, and CIFAR loading for the experiment notebooks."""

import contextlib

import numpy as np
import torch
import torch.nn as nn


def iterate_batches(x: torch.Tensor, y: torch.Tensor, bs: int, shuffle: bool = True):
    """Yield paired batches on the tensors' device, including the final partial batch."""
    n = x.size(0)
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for start in range(0, n, bs):
        b = idx[start : start + bs]
        yield x[b], y[b]


def train_epoch(
    model: nn.Module,
    data: tuple,
    opt: torch.optim.Optimizer,
    crit: nn.Module,
    bs: int,
    grad_clip: float | None = None,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Train once over the data and return mean loss and accuracy in percent.

    The criterion must average over its batch. Model and data must share a
    device. Set grad_clip to clip the gradient norm, and amp_dtype to enable
    mixed precision on CUDA; CPU computation remains full precision.
    """
    model.train()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None and x.device.type == "cuda"
        else contextlib.nullcontext()
    )

    for xb, yb in iterate_batches(x, y, bs):
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(xb)
            loss = crit(out, yb)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: tuple,
    crit: nn.Module,
    bs: int,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Return sample-weighted mean loss and accuracy in percent, without gradients."""
    model.eval()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None and x.device.type == "cuda"
        else contextlib.nullcontext()
    )

    for xb, yb in iterate_batches(x, y, bs, shuffle=False):
        with amp_ctx:
            out = model(xb)
            loss = crit(out, yb)
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)

    return loss_sum / total, 100.0 * correct / total


