import os
import argparse
from datetime import datetime
from pathlib import Path
import warnings


import numpy as np
import torch
from tqdm import trange, tqdm
from torch import nn

from fmri_mapping.io.nsd import get_resource
from fmri_mapping.utils.logger import WandbLogger
from fmri_mapping.embedding.ops import (
    split_betas_by_subject_index,
    split_repetitions,
    to_tensor,
)
from fmri_mapping.embedding.evaluation import (
    compute_metric_curves,
    avg_metric_combinations,
    accuracy_cosine_similarity,
    apply_batched_model,
)

from fmri_mapping.embedding.losses import (
    info_nce_symmetric,
    geometry_cosine_reg,
    pull_cosine,
    circle_loss,
    vsp_loss,
    loo_repeat_denoise_loss,
)
from fmri_mapping.embedding.mlp import ResidualMLP
from fmri_mapping.utils import get_device


WANDB_MODE = "disabled"  # "offline", "online" or "disabled"
WANDB_PROJECT = "fmri-mapping-encoder"
os.environ["WANDB_SILENT"] = "true"


EVALUATION_BATCH_SIZE = 8192


def compute_losses(
    Z1,
    Z2,
    lambda_nce: float = 1.0,
    temperature: float = 0.09,
    lambda_geom: float = 0.1,
    lambda_pull: float = 0.5,
    lambda_identity_preserve: float = 0.0,
    lambda_circle: float = 0.0,
    lambda_vsp: float = 0.0,
    lambda_mse: float = 0.0,
):
    total_loss = torch.tensor(0.0, device=Z1.device)
    stats = {}

    if lambda_nce > 0:
        loss_info_nce = info_nce_symmetric(Z1, Z2, temperature=temperature)
        total_loss = total_loss + lambda_nce * loss_info_nce
        stats["loss_info_nce"] = loss_info_nce.item()
    if lambda_geom > 0:
        loss_cosine = geometry_cosine_reg(Z1, Z2)
        total_loss = total_loss + lambda_geom * loss_cosine
        stats["loss_geom_cosine"] = loss_cosine.item()
    if lambda_pull > 0:
        loss_pull_cosine = pull_cosine(Z1, Z2)
        total_loss = total_loss + lambda_pull * loss_pull_cosine
        stats["loss_pull_cosine"] = loss_pull_cosine.item()
    if lambda_identity_preserve > 0:
        loss_identity = 0.5 * (pull_cosine(Z1, Z1) + pull_cosine(Z2, Z2))
        total_loss = total_loss + lambda_identity_preserve * loss_identity
        stats["loss_identity_preserve"] = loss_identity.item()
    if lambda_circle > 0:
        loss_circle = circle_loss(Z1, Z2)
        total_loss = total_loss + lambda_circle * loss_circle
        stats["loss_circle"] = loss_circle.item()
    if lambda_vsp > 0:
        loss_vsp = vsp_loss(Z1, Z2)
        total_loss = total_loss + lambda_vsp * loss_vsp
        stats["loss_vsp"] = loss_vsp.item()
    if lambda_mse > 0:
        loss_mse = nn.functional.mse_loss(Z1, Z2, reduction="sum")
        total_loss = total_loss + lambda_mse * loss_mse
        stats["loss_mse"] = loss_mse.item()

    return total_loss, stats


def _add_keys_prefix(d: dict, prefix: str) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def train_mlp_triplets(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    Z3: torch.Tensor,
    model: nn.Module,
    decoder_model: nn.Module,
    temperature: float = 0.07,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer: str = "adamw",
    batch_size: int = 512,
    steps: int = 2000,
    eval_every: int = 200,
    device: str = "cuda",
    seed: int = 42,
    lambda_nce: float = 1.0,
    lambda_geom: float = 0.01,
    lambda_pull: float = 0.0,
    lambda_identity_preserve: float = 0.0,
    lambda_circle: float = 0.0,
    lambda_vsp: float = 0.0,
    lambda_loo: float = 0.0,
    lambda_mse: float = 0.1,
    milestones: list[int] = [1300, 1400, 1500, 1600, 1700, 1800, 1900],
    gamma: float = 0.3,
    logger: WandbLogger = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert Z1.shape == Z2.shape == Z3.shape
    N, D = Z1.shape

    model = model.to(device)
    decoder_model = decoder_model.to(device)
    model.train()
    decoder_model.train()

    # Reproduction: explicit manuscript optimizer.
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer]
    opt = optimizer_class(
        list(model.parameters()) + list(decoder_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=milestones,  # can add more, e.g. [1500, 1800]
        gamma=gamma,  # lr <- lr * 0.1 at each milestone
    )

    Z1 = Z1.to(device)
    Z2 = Z2.to(device)
    Z3 = Z3.to(device)
    pairs = [(Z1, Z2), (Z1, Z3), (Z2, Z3)]
    stats = []

    postfix = {}
    postfix["loss"] = f"{0:.3f}"
    postfix["alpha"] = f"{model.alpha.item():.2f}"
    postfix["acc"] = f"{0:.3f}"
    postfix["rank"] = f"{0:.2f}"

    model.train()
    loss_kwargs = dict(
        lambda_nce=lambda_nce,
        lambda_geom=lambda_geom,
        lambda_pull=lambda_pull,
        lambda_identity_preserve=lambda_identity_preserve,
        lambda_circle=lambda_circle,
        lambda_vsp=lambda_vsp,
        lambda_mse=lambda_mse,
        temperature=temperature,
    )

    for step in (pbar := trange(steps, desc="Training MLP", position=1, leave=False)):
        opt.zero_grad()

        idx = torch.randint(0, N, (batch_size,), device=device)
        Zal, Zbl, Zcl = model(Z1[idx]), model(Z2[idx]), model(Z3[idx])
        loss = 0.0
        for Zal, Zbl in [(Zal, Zbl), (Zal, Zcl), (Zbl, Zcl)]:
            encode_losses, encode_stats = compute_losses(Zal, Zbl, **loss_kwargs)
            logger.logkvs(encode_stats)
            loss = loss + encode_losses

        loss = loss / len(pairs)

        if lambda_loo > 0:
            loss_loo = loo_repeat_denoise_loss(Z1[idx], Z2[idx], Z3[idx])
            loss = loss + lambda_loo * loss_loo
            logger.logkv("loss_loo", loss_loo.item())

        logger.logkvs({"loss_total": loss.item(), "alpha": model.alpha.item()})

        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        logger.logkv("grad_norm", norm.item())
        opt.step()
        scheduler.step()
        logger.dumpkvs()
        postfix["loss"] = f"{loss.item():.3f}"
        postfix["alpha"] = f"{model.alpha.item():.2f}"
        pbar.set_postfix(postfix)

    return model, stats

