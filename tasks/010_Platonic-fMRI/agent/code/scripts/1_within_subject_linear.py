import argparse
from typing import Literal
import torch
import os
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from sklearn.decomposition import PCA
from mvlearn.embed import MCCA

from tqdm import tqdm

from fmri_mapping.embedding.evaluation import (
    compute_metric_curves,
    avg_metric_combinations,
)

from fmri_mapping.embedding.linear import (
    fit_session_residualizer,
    voxel_reliability_loo,
    distill_mcca_projector,
    estimate_sample_weights,
)
from fmri_mapping.embedding.ops import (
    get_subject_betas,
    get_confound_matrix,
    split_betas_by_subject_index,
    split_repetitions,
)

from fmri_mapping.io.nsd import get_resource
from fmri_mapping.embedding.evaluation import compute_metric_curves



def compute_linear_within_subject_pipeline(
    subject: int,
    betas: np.ndarray,
    df_stimuli: pd.DataFrame,
    df_repetitions_train: pd.DataFrame,
    confound_reg: float = 0.0,
    reliability_eps: float = 1e-6,
    n_components_pca: int = 512,
    n_components_mcca: int = 128,
    reg_mcca: float = 0.1,
    mcca_inpute="mean",
    mcca_distill_reg: float = 0.5,
    progress: bool = True,
    apply_sample_weights: bool = True,
    skip_residualization: bool = False,
    skip_reliability: bool = False,
    skip_mcca: bool = False,
    skip_mcca_distill: bool = False,
):
    """Compute within-subject linear embedding pipeline."""
    memory_opt = os.environ.get("REPRO_LINEAR_MEMORY") == "1"
    if memory_opt:
        from fmri_mapping.embedding.cpu_linear import (
            reliability_blocked, sample_weights_blocked,
            weighted_training_copy, project_blocked)

    pbar = tqdm(
        total=5,
        desc=f"Subject {subject}",
        disable=not progress,
        position=1,
        leave=False,
    )

    train_mask = (df_stimuli.shared == False).to_numpy(
        dtype=bool
    )  # Should be all true. But for safety.

    pbar.set_description(f"Residualizing confounds")
    # Residualize betas with session_run confound
    
    C = get_confound_matrix(subject)  # n_trials x n_confounds
    B = fit_session_residualizer(
        betas[train_mask, :], C[train_mask, :], lam=confound_reg
    )
    if skip_residualization:
        betas_res = betas
    else:
        if memory_opt:
            betas_res = betas
            betas_res -= C @ B
        else:
            betas_res = betas - C @ B  # n_trials x n_voxels
        del betas

    gc.collect()
    pbar.update(1)
    
    

    #Compute reliability mask

    
    if memory_opt:
        W_rel = reliability_blocked(betas_res, df_repetitions_train)
    else:
        W_rel = voxel_reliability_loo(
            *split_betas_by_subject_index(betas_res, df_repetitions_train, inpute=False)
        )

    W_rel = np.clip(W_rel, 0.0, None) + reliability_eps  # n_voxels

    if skip_reliability:
        betas_rel = betas_res
    else:
        if memory_opt:
            betas_rel = betas_res
            betas_rel *= W_rel[None, :]
        else:
            betas_rel = W_rel[None, :] * betas_res  # n_trials x n_voxels
        del betas_res

    gc.collect()
    pbar.update(1)

    # # Apply PCA
    pbar.set_description(f"Fitting PCA")
    pca = PCA(n_components=n_components_pca, random_state=int(os.environ.get("BENCH_SEED", "42")), copy=not memory_opt)

    if apply_sample_weights:
        weight_function = sample_weights_blocked if memory_opt else estimate_sample_weights
        W_sample = weight_function(betas_rel, df_repetitions_train, train_mask)
        W_sample = np.clip(W_sample, 1e-3, None)  # Avoid zero weights
        W_sample = np.sqrt(W_sample)  # Convert to sample weights for PCA
        if memory_opt:
            pca.fit(weighted_training_copy(betas_rel, train_mask, W_sample))
        else:
            pca.fit(W_sample[:, None] * betas_rel[train_mask, :])  # Fit on train
    else:
        pca.fit(betas_rel[train_mask, :])  # Fit on train

    W_pca = pca.components_.T  # n_voxels x n_pca_components
    mu_pca = pca.mean_  # n_voxels

    # Apply PCA
    if memory_opt:
        betas_pca = project_blocked(betas_rel, mu_pca, W_pca)
    else:
        betas_pca = (betas_rel - mu_pca[None, :]) @ W_pca  # n_trials x n_pca_components
    del betas_rel
    gc.collect()
    pbar.update(1)

    # Fit MCCA
    pbar.set_description(f"Fitting MCCA")
    mcca = MCCA(n_components=n_components_mcca, regs=reg_mcca)
    X1, X2, X3 = split_betas_by_subject_index(
        betas_pca, df_repetitions_train, inpute=mcca_inpute
    )  # Fit on train
    mcca.fit([X1, X2, X3])
    pbar.update(1)

    # Distill a unique projector with ridge regression
    pbar.set_description(f"Distilling MCCA projector")
    
    if skip_mcca:
        Z = betas_pca
        matrices = {}
        return Z, matrices
    
    if skip_mcca_distill:
        W_mcca, b_mcca = distill_mcca_projector(
        betas_pca[train_mask, :], mcca, lam_ridge=mcca_distill_reg, target="single_view"
    )
    else:
        W_mcca, b_mcca = distill_mcca_projector(
            betas_pca[train_mask, :], mcca, lam_ridge=mcca_distill_reg
        )

    # Inference on all data
    
    
    Z = (betas_pca @ W_mcca) + b_mcca[None, :]  # n_trials x n_mcca_components
    

    matrices = {
        "C": C,  # n_trials x n_confounds
        "B": B,  # n_confounds x n_voxels
        "W_rel": W_rel,  # n_voxels
        "W_pca": W_pca,  # n_voxels x n_pca_components
        "mu_pca": mu_pca,  # n_voxels
        "W_mcca": W_mcca,  # n_pca_components x n_mcca_components
        "b_mcca": b_mcca,  # n_mcca_components
    }
    pbar.update(1)

    return Z, matrices

