"""Fixed internal fit/calibration partition of training data; never reads validation."""
from pathlib import Path
import numpy as np

SPLIT_SEED = 20261102


def internal_split(n):
    if n < 2:
        raise ValueError('at least two training observations required')
    # Match a 20% held-out calibration fraction, with stable original row order.
    n_cal = max(1, int(np.ceil(0.2 * n)))
    order = np.random.RandomState(SPLIT_SEED).permutation(n)
    return np.sort(order[n_cal:]), np.sort(order[:n_cal])


def load_training_data(data, datasets):
    states, counts = {}, {}
    with np.load(Path(data) / 'train/train.npz', allow_pickle=False) as archive:
        for name in datasets:
            x, y = archive['x_' + name], archive['y_' + name]
            if len(x) != len(y):
                raise ValueError('training features/targets are misaligned')
            fit, cal = internal_split(len(y))
            # Existing estimator routines call their selection/calibration
            # arrays xv/yv. These are internal training rows, never data/val.
            states[name] = {'xt': x[fit], 'yt': y[fit], 'xv': x[cal], 'yv': y[cal]}
            counts[name] = {'fit': len(fit), 'calibration': len(cal), 'total_training': len(y)}
    return states, counts
