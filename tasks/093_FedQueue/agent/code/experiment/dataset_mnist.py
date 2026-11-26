"""Offline, materialized training data. Validation is used only by selection."""
from pathlib import Path
import os
import numpy as np
import torch
from appfl.misc.data import Dataset

def get_mnist(num_clients, client_id, data_root, partition_path, **kwargs):
    assert num_clients == 4
    with np.load(Path(data_root) / 'train' / 'train.npz', allow_pickle=False) as z:
        with np.load(partition_path, allow_pickle=False) as parts:
            indices = parts[f'client_{client_id}']
        inputs = torch.from_numpy(z['x'][indices]).unsqueeze(1).float().div(255)
        labels = torch.from_numpy(z['y'][indices].astype(np.int64))
    return Dataset(inputs, labels), None
