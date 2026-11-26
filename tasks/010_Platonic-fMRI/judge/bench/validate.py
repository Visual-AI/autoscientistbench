#!/usr/bin/env python3
"""Validate finite per-trial embedding tensors and exact sorted trial ids."""
import sys
from pathlib import Path
import numpy as np

def read_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    subjects = sorted(int(k.removeprefix('id_')) for k in arrays if k.startswith('id_'))
    if not subjects or set(arrays) != {f'{prefix}_{s}' for s in subjects for prefix in ['id','embedding']}:
        raise ValueError('Expected exactly id_s and embedding_s for each subject')
    for s in subjects:
        ids, z = arrays[f'id_{s}'], arrays[f'embedding_{s}']
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0 or np.any(ids[1:] <= ids[:-1]):
            raise ValueError('ids must be nonempty, integer, unique and sorted')
        if z.ndim != 2 or z.shape[0] != len(ids) or not 1 <= z.shape[1] <= 128 or z.dtype.kind != 'f' or not np.isfinite(z).all():
            raise ValueError('embeddings must be finite floating N by D matrices with 1 <= D <= 128')
    return arrays, subjects

def validate(pred_path, input_path):
    arrays, subjects = read_prediction(pred_path)
    paths = sorted(Path(input_path).glob('subject_*.npz'))
    expected = sorted(int(p.stem.split('_')[-1]) for p in paths)
    if subjects != expected:
        raise ValueError('subject set mismatch')
    for path in paths:
        subject = int(path.stem.split('_')[-1])
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data['id'], arrays[f'id_{subject}']):
                raise ValueError('trial id mismatch or wrong order')
    return arrays, subjects

if __name__ == '__main__':
    try:
        _, subjects = validate(*sys.argv[1:])
        print(f'ok: {len(subjects)} subjects')
    except Exception as exc:
        sys.exit(f'invalid prediction: {exc}')
