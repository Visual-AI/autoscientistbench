"""Original Table-2 geometric width objective and per-cell coverage guards."""
import math
import numpy as np

CELLS = tuple(d + '_' + f for d in ('Amazon', 'Google', 'Electricity', 'Temperature')
              for f in ('Prophet', 'AR', 'Theta'))
FORMAL_SIZES = {'Amazon': (3020, 100), 'Google': (3020, 100),
                'Electricity': (2000, 300), 'Temperature': (1575, 500)}


def load_replay(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    cells = values['cells'].tolist()
    if len(cells) != 12 or set(cells) != set(CELLS):
        raise ValueError('all twelve original cells required')
    flag = values['synthetic_development']
    if flag.shape != () or flag.dtype.kind != 'b':
        raise ValueError('missing development designation')
    synthetic = bool(flag)
    expected = {'cells', 'synthetic_development'} | {c + '__' + k for c in CELLS
                                                     for k in ('y', 'forecast', 'burn')}
    if set(values) != expected:
        raise ValueError('unexpected replay fields')
    result = {}
    for cell in CELLS:
        y, forecast, burn_value = (values[cell + '__' + key] for key in ('y', 'forecast', 'burn'))
        if (y.ndim != 1 or y.shape != forecast.shape or y.dtype != np.dtype('float64')
                or forecast.dtype != y.dtype or not np.isfinite(y).all()
                or not np.isfinite(forecast).all() or burn_value.shape != ()
                or burn_value.dtype.kind not in 'iu'):
            raise ValueError('invalid replay arrays')
        burn = int(burn_value)
        if not 0 < burn < len(y) - 1:
            raise ValueError('invalid burn or empty evaluation slice')
        if not synthetic and (len(y), burn) != FORMAL_SIZES[cell.rsplit('_', 1)[0]]:
            raise ValueError('formal timeline changed')
        result[cell] = {'y': y, 'forecast': forecast, 'burn': burn}
    return result, synthetic


def cell_metrics(y, lower, upper, burn):
    if y.ndim != 1 or y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError('interval shape mismatch')
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper)):
        raise ValueError('invalid interval bounds')
    valid = slice(burn + 1, None)
    n_eval = len(y[valid])
    if n_eval < 1:
        raise ValueError('empty scoring slice')
    with np.errstate(over='ignore', invalid='ignore'):
        widths = upper - lower
        mean_width = float(np.mean(widths[valid]))
    coverage = float(np.mean(((lower <= y) & (y <= upper))[valid]))
    threshold = .9 - 2 * math.sqrt(.09 / n_eval)
    passed = (np.isfinite(widths).all() and math.isfinite(mean_width)
              and coverage >= threshold)
    return {'n_eval': n_eval, 'mean_width': mean_width, 'coverage': coverage,
            'coverage_threshold': threshold, 'passed': bool(passed)}


def aggregate(groups):
    if set(groups) != set(CELLS):
        raise ValueError('missing cells')
    widths = [groups[cell]['mean_width'] for cell in CELLS]
    valid = all(math.isfinite(w) and w >= 0 for w in widths)
    passed = valid and all(groups[cell]['passed'] for cell in CELLS)
    value = (0. if any(w == 0 for w in widths) else
             math.exp(math.fsum(math.log(w) for w in widths) / 12)) if valid else math.inf
    return {'geomean_width': value if passed else math.inf,
            'all_constraints_pass': passed, 'cells': groups}
