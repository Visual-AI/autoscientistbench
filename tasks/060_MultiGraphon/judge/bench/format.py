"""Frozen array format helpers; pickle is never accepted."""
import numpy as np


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_npz(path, **arrays):
    with open(path, 'wb') as f:
        np.savez_compressed(f, **arrays)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_prediction(path, expected_ids=None):
    p = read_npz(path)
    need(set(p) == {'id', 'k', 'offsets', 'values', 'time_s'}, 'wrong prediction keys')
    ids, k, offsets, values, seconds = (p[x] for x in ('id','k','offsets','values','time_s'))
    need(ids.ndim == 1 and ids.dtype.kind in 'US' and len(ids) > 0, 'id must be a nonempty string vector')
    n = len(ids)
    need(len(set(ids.tolist())) == n, 'duplicate ids')
    if expected_ids is not None:
        need(set(ids.tolist()) == set(expected_ids), 'id set mismatch')
    need(k.shape == (n,) and k.dtype.kind in 'iu', 'invalid k')
    need(np.all((k >= 1) & (k <= 1000)), 'k must lie in 1..1000')
    need(offsets.shape == (n+1,) and offsets.dtype.kind in 'iu', 'invalid offsets')
    need(offsets[0] == 0 and np.array_equal(np.diff(offsets), k.astype(np.int64)**2), 'offsets must delimit k squared entries')
    need(values.ndim == 1 and len(values) == offsets[-1], 'wrong number of matrix entries')
    need(values.dtype.kind == 'f' and np.all(np.isfinite(values)), 'matrix entries must be finite floats')
    need(np.all((values >= 0) & (values <= 1)), 'probabilities must lie in [0,1]')
    need(seconds.shape == (n,) and seconds.dtype.kind == 'f', 'invalid time_s')
    need(np.all(np.isfinite(seconds)) and np.all(seconds > 0), 'times must be finite and positive')
    out = {}
    for i, iid in enumerate(ids):
        w = values[offsets[i]:offsets[i+1]].reshape(int(k[i]), int(k[i]))
        need(np.allclose(w, w.T, atol=1e-12, rtol=0), 'graphon must be symmetric')
        out[str(iid)] = (w, float(seconds[i]))
    return out


def unpack_collection(obs, i):
    flat = obs['adjacency'][obs['offsets'][i]:obs['offsets'][i+1]]
    graphs, j = [], 0
    for n in obs['vec_n'][i]:
        n = int(n)
        graphs.append(flat[j:j+n*n].reshape(n, n).astype(int))
        j += n*n
    need(j == len(flat), 'bad adjacency packing')
    return graphs
