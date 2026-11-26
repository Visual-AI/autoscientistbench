"""Strict binary prediction validation. Usage: validate.py PRED INPUT_DIR."""
import sys
from pathlib import Path
import numpy as np


def check(pred_path, input_path):
    paths = sorted(Path(input_path).glob('*.npz'))
    assert paths, 'empty input'
    expected = {key for p in paths for key in (p.stem, p.stem + '__calls')}
    with np.load(pred_path, allow_pickle=False) as pred:
        assert len(pred.files) == len(set(pred.files)), 'duplicate archive keys'
        assert set(pred.files) == expected, 'prediction id/key mismatch'
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                dim = int(data['dim'])
            arr = pred[path.stem]
            assert arr.shape == (10000, dim), 'sample shape'
            assert arr.dtype.kind == 'f' and np.isfinite(arr).all(), 'nonfinite or nonfloat samples'
            calls = pred[path.stem + '__calls']
            assert calls.shape == () and calls.dtype.kind in 'iu' and int(calls) >= 0, 'invalid call count'
    return len(paths)


if __name__ == '__main__':
    try:
        print(f'ok: {check(sys.argv[1], sys.argv[2])} instances')
    except Exception as exc:
        sys.exit(f'invalid prediction: {exc}')
