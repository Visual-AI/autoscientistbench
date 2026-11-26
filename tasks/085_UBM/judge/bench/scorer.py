#!/usr/bin/env python3
"""Frozen score rule. FP constraint failure => 0; invalid certificate => Infinity."""
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import read_prediction, statistics, SIGNALS

METRIC = 'rho_b1_A_detection'
GUARDS = [('FP_A_no_bias', 'no_bias', 'A', .04),
          ('FP_A_selection_type1', 'selection_bias', 'A', .087),
          ('FP_A_transportability', 'transportability_bias', 'A', .160),
          ('FP_S_confounding', 'confounding_bias', 'S', .142)]


def aggregate(stats, mechanisms):
    rates = {}
    for mechanism in sorted(set(mechanisms)):
        subset = stats[np.asarray(mechanisms) == mechanism].reshape(-1, 4, 2)
        rates[mechanism] = {key: float(np.mean((subset[:, j, 0] > 0) & (subset[:, j, 1] < .01)))
                            for j, key in enumerate(SIGNALS)}
    guards = {name: {'value': rates[m][s], 'maximum': cap, 'passed': rates[m][s] <= cap}
              for name, m, s, cap in GUARDS}
    raw = rates['confounding_bias']['A']
    ok = all(g['passed'] for g in guards.values())
    return {METRIC: raw if ok else 0.0, 'raw_detection': raw, 'guards_ok': ok,
            'guards': guards, 'rates': rates}


def score(pred_path, labels_path):
    with np.load(labels_path, allow_pickle=False) as z:
        ids = z['id'].astype(str).tolist()
        pred = read_prediction(pred_path, ids)
        observations = z['eval']
        recomputed = np.asarray([statistics(pred[i][:256], observations[j]) for j, i in enumerate(ids)])
        reported = np.asarray([pred[i][256:] for i in ids])
        if not np.isfinite(recomputed).all():
            return {METRIC: float('inf'), 'invalid': 'undefined Pearson correlation'}
        # Same arithmetic as full patient prediction; tolerance covers BLAS roundoff only.
        error = float(np.max(np.abs(recomputed - reported)))
        mismatch = not np.allclose(recomputed, reported, atol=2e-7, rtol=2e-7)
        decisions_differ = np.any(((recomputed[:, ::2] > 0) & (recomputed[:, 1::2] < .01)) !=
                                  ((reported[:, ::2] > 0) & (reported[:, 1::2] < .01)))
        result = aggregate(recomputed, z['mechanism'].astype(str))
        result.update(certificate_max_abs_error=error, certificate_ok=not (mismatch or decisions_differ))
        if mismatch or decisions_differ:
            result[METRIC] = float('inf')
            result['invalid'] = 'reported statistics do not match probability certificate'
        return result


if __name__ == '__main__':
    try:
        output = score(*sys.argv[1:])
    except Exception as e:
        output = {METRIC: float('inf'), 'invalid': str(e)}
    print(json.dumps(output))
