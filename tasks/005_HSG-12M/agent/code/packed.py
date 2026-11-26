import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class IndexDataset(Dataset):
    def __init__(self, indices): self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return int(self.indices[i])

class PackedCollator:
    def __init__(self, path, labels=None):
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {k[5:]: z[k] for k in z.files if k.startswith('data_')}
            self.ptrs = {k[4:]: z[k] for k in z.files if k.startswith('ptr_')}
            self.ids = z['id']
        if labels is not None:
            import csv
            with open(labels, newline='') as f:
                truth = {row['id']: int(row['y']) for row in csv.DictReader(f)}
            self.arrays['y'] = np.array([truth[i] for i in self.ids], dtype=np.int64)
            self.ptrs['y'] = np.arange(len(self.ids)+1)

    @staticmethod
    def gather_ranges(ptr, ids):
        starts = ptr[ids]
        lengths = ptr[ids+1] - starts
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        index = np.arange(offsets[-1]) + np.repeat(starts-offsets[:-1], lengths)
        return index, lengths, offsets

    def __call__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        ni, nn, nptr = self.gather_ranges(self.ptrs['x'], ids)
        ei, ne, _ = self.gather_ranges(self.ptrs['edge_index'], ids)
        result = {}
        for key, a in self.arrays.items():
            if key == 'edge_index':
                value = a[:, ei] + np.repeat(nptr[:-1], ne)[None, :]
            elif np.array_equal(self.ptrs[key], self.ptrs['x']): value = a[ni]
            elif np.array_equal(self.ptrs[key], self.ptrs['edge_index']): value = a[ei]
            else:
                gi, _, _ = self.gather_ranges(self.ptrs[key], ids)
                value = a[gi]
            result[key] = torch.from_numpy(np.ascontiguousarray(value))
        result['batch'] = torch.from_numpy(np.repeat(np.arange(len(ids)), nn))
        result['ptr'] = torch.from_numpy(nptr)
        batch = Batch(**result)
        batch._num_graphs = len(ids)
        return batch

