"""Lossless checkpoints for fixed-seed QRF fits; opt-in via paper-local env."""
import gzip
import hashlib
import importlib.metadata
import os
import pickle
from pathlib import Path
import tempfile
import numpy as np


def cache_path(X, y, models, seed, index, source):
    directory = os.environ.get('CLEAR_QRF_CACHE')
    if not directory or not isinstance(seed, (int, np.integer)):
        return None
    root = Path(os.environ['PAPER_DIR']).resolve()
    directory = Path(directory).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('CLEAR_QRF_CACHE must be inside PAPER_DIR')
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'scikit-learn', 'quantile-forest', 'joblib']}
    h = hashlib.sha256()
    h.update(Path(source).read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(pickle.dumps((versions, [m.get_params() for m in models], int(seed), int(index)), protocol=5))
    for array in [X, y]:
        array = np.ascontiguousarray(array)
        h.update(pickle.dumps((array.shape, array.dtype.str), protocol=5))
        h.update(array.tobytes())
    signature = h.hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (signature + '.pkl.gz')


def read(path):
    if path is None or os.environ.get('CLEAR_QRF_CACHE_RESUME') != '1' or not path.exists():
        return None
    with gzip.open(path, 'rb') as stream:
        value = pickle.load(stream)
    if value['signature'] != path.stem.removesuffix('.pkl'):
        raise ValueError(f'QRF cache signature mismatch: {path}')
    np.random.set_state(value['rng'])
    return value['models']


def write(path, models):
    if path is None:
        return
    payload = dict(signature=path.stem.removesuffix('.pkl'), models=models, rng=np.random.get_state())
    fd, name = tempfile.mkstemp(prefix='.qrf-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
                pickle.dump(payload, stream, protocol=5)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
