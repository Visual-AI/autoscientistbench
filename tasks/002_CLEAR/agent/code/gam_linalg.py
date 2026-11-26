"""Equivalent economy SVD for pygam PIRLS, with a robust LAPACK fallback.

pygam only reads U[:min_n_m,:min_n_m]. The unused orthogonal complement is
unnecessary. This does not change its basis, penalty, tolerance or iterations.
"""
import inspect
import textwrap
import hashlib
from pathlib import Path
import numpy as np
import scipy.linalg
import pygam.pygam as pg
ORIGINAL=pg.GAM._pirls
DIRECTORY=None

def economy_svd(a):
    try:
        return np.linalg.svd(a,full_matrices=False)
    except np.linalg.LinAlgError:
        # Retry the same finite matrix with the classical LAPACK driver.
        assert np.isfinite(a).all()
        if DIRECTORY is not None:
            directory=Path(DIRECTORY);directory.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(a.tobytes()).hexdigest()
            np.save(directory/(digest+'.npy'),a)
        return scipy.linalg.svd(a,full_matrices=False,lapack_driver='gesvd')

def install(directory=None):
    global DIRECTORY
    DIRECTORY=directory
    if pg.GAM._pirls is not ORIGINAL:return
    source=textwrap.dedent(inspect.getsource(ORIGINAL))
    old='np.linalg.svd(np.vstack([R, E]))'
    assert source.count(old)==1
    source=source.replace(old,'_opt30_svd(np.vstack([R, E]))')
    scope=dict(ORIGINAL.__globals__,_opt30_svd=economy_svd)
    exec(compile(source,__file__,'exec'),scope)
    pg.GAM._pirls=scope['_pirls']

def restore():pg.GAM._pirls=ORIGINAL
