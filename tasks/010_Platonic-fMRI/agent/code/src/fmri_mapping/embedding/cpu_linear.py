"""Bounded-memory implementations of the unchanged within-subject estimators."""
import numpy as np
from .linear import corr_cols, voxel_reliability_loo
from .ops import split_betas_by_subject_index


def reliability_blocked(betas, repetitions, block=1024):
    result=np.empty(betas.shape[1],dtype=betas.dtype)
    for start in range(0,betas.shape[1],block):
        end=min(start+block,betas.shape[1])
        result[start:end]=voxel_reliability_loo(*split_betas_by_subject_index(
            betas[:,start:end],repetitions,inpute=False))
    return result


def sample_weights_blocked(X_train, repetitions, train_mask, eps=1e-8, block=64):
    # Missing views have all-NaN rows in the original. No imputation is used
    # for this estimator. Full feature vectors are retained in each block.
    indexes=[repetitions[f'subject_index_{j+1}'].fillna(-1).to_numpy(dtype=np.int64)
             for j in range(3)]
    sums=np.zeros(len(repetitions));counts=np.zeros(len(repetitions))
    for a,b in [(0,1),(0,2),(1,2)]:
        valid=np.flatnonzero((indexes[a]>=0)&(indexes[b]>=0))
        for start in range(0,len(valid),block):
            rows=valid[start:start+block]
            A=X_train[indexes[a][rows]];B=X_train[indexes[b][rows]]
            finite=~np.isnan(A).any(axis=1)&~np.isnan(B).any(axis=1)
            rows=rows[finite]
            values=corr_cols(A[finite].T,B[finite].T)
            sums[rows]+=values;counts[rows]+=1
    weights=sums/(counts+eps)
    scattered=np.full(len(X_train),np.nan)
    for indexes_j in indexes:
        valid=indexes_j>=0
        scattered[indexes_j[valid]]=weights[valid]
    scattered=scattered[train_mask]
    return np.where(np.isnan(scattered),np.nanmedian(scattered),scattered)


def weighted_training_copy(betas, train_mask, weights, block=256):
    indexes=np.flatnonzero(train_mask)
    result=np.empty((len(indexes),betas.shape[1]),dtype=np.result_type(betas,weights))
    for start in range(0,len(indexes),block):
        end=min(start+block,len(indexes))
        np.multiply(weights[start:end,None],betas[indexes[start:end]],out=result[start:end])
    return result


def project_blocked(betas, mean, weights, block=256):
    result=np.empty((len(betas),weights.shape[1]),dtype=np.result_type(betas,mean,weights))
    for start in range(0,len(betas),block):
        end=min(start+block,len(betas))
        result[start:end]=(betas[start:end]-mean[None,:])@weights
    return result
