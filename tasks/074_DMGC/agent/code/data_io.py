import os
from pathlib import Path
import numpy as np
import torch
from utils import normalize_features

def load_graph():
    with np.load(Path(os.environ['BENCH_DATA_DIR'])/'train'/'graph.npz',allow_pickle=False) as d:
        ids=d['ids'].copy()
        feature=torch.FloatTensor(d['features'].copy())
        adj=torch.FloatTensor(d['adjacency'].copy())
    # Same tensors as the released Wisconsin loader, including duplicated views.
    return ids, [adj,adj.clone()], normalize_features(feature), feature, 2
