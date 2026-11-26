"""Run the unchanged APPFL partitioner using lightweight dummy features.
Partition indices depend only on labels, in their original order.
"""
import numpy as np
import torch
from appfl.misc.data import dirichlet_noniid_partition

class LabelDataset:
    def __init__(self, labels): self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        if i >= len(self.labels): raise IndexError(i)
        return torch.zeros(1), int(self.labels[i])

def partition_indices(labels, seed):
    parts = dirichlet_noniid_partition(LabelDataset(labels), 4, alpha1=8, alpha2=.1, seed=seed)
    return {f'client_{i}': p.sample_indices for i, p in enumerate(parts)}
