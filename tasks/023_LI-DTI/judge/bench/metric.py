"""Frozen CSV contract and exact tie-aware PR integral; standard library only."""
import csv
import math
from collections import defaultdict

KEYS = ['run', 'fold', 'sample', 'row', 'col']

def read_rows(path, value=None):
    result = {}
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != KEYS + ([value] if value else []):
            raise ValueError('unexpected CSV header')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('wrong number of columns')
            key = tuple(int(row[k]) for k in KEYS)
            if any(str(v) != row[k] or v < 1 for k, v in zip(KEYS, key)):
                raise ValueError('coordinates must be positive canonical integers')
            if key in result:
                raise ValueError('duplicate coordinate')
            if value == 'pred':
                v = float(row[value])
                if not math.isfinite(v):
                    raise ValueError('nonfinite prediction')
            elif value == 'y':
                if row[value] not in ('0', '1'):
                    raise ValueError('label must be 0 or 1')
                v = int(row[value])
            else:
                v = None
            result[key] = v
    if not result:
        raise ValueError('empty file')
    return result

def align(pred, expected):
    if pred.keys() != expected.keys():
        raise ValueError('missing or extra coordinates')

def aupr(labels, scores):
    """AUPR_perfcurve.m: last point per tied score; no artificial (0,1)."""
    total_positive = sum(labels)
    if total_positive == 0:
        raise ValueError('no positive labels')
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    points = []
    for index, (score, label) in enumerate(ordered):
        tp += label
        fp += 1 - label
        if index + 1 == len(ordered) or ordered[index + 1][0] != score:
            points.append((tp / total_positive, tp / (tp + fp)))
    return math.fsum((r2-r1)*(p1+p2)/2 for (r1,p1),(r2,p2) in zip(points, points[1:]))

def evaluate(pred, labels):
    align(pred, labels)
    groups = defaultdict(list)
    for key, y in labels.items():
        groups[key[:2]].append((y, pred[key]))
    # A full input contains the original10x10 groups; only the explicit smoke
    # labels contain one group. Exact keys above prohibit candidate truncation.
    valid_groups = {(r,f) for r in range(1,11) for f in range(1,11)}
    if set(groups) not in (valid_groups, {(1,1)}):
        raise ValueError('expected10x10 groups, or explicit one-fold smoke labels')
    by_run = defaultdict(list)
    fold_scores = {}
    for (run, fold), rows in sorted(groups.items()):
        ys, ps = zip(*rows)
        if not 0 < sum(ys) < len(ys):
            raise ValueError('fold must contain both classes')
        value = aupr(ys, ps)
        fold_scores[f'{run}:{fold}'] = value
        by_run[run].append(value)
    means = [math.fsum(v)/len(v) for _,v in sorted(by_run.items())]
    mean = math.fsum(means)/len(means)
    std = math.sqrt(math.fsum((x-mean)**2 for x in means)/(len(means)-1)) if len(means)>1 else 0.0
    return {'AUPR':mean,'AUPR_std':std,'run_means':means,'fold_AUPR':fold_scores,'n_pairs':len(labels),'n_fits':len(groups)}
