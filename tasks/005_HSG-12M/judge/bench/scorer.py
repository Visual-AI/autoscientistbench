#!/usr/bin/env python3
"""Paper Eq. (22) overall Top-1 accuracy under the fixed capacity and training-work controls.
The scorer checks declarations. Their truth requires the model probe and patch audit.
"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction import read_predictions

BASELINE_PARAMETERS = 703080
PARAMETER_LIMIT = BASELINE_PARAMETERS * 104 // 100

def score(pred, labels):
    rows, (parameters, steps, epochs, digest) = read_predictions(pred)
    with open(labels, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id','y']: raise ValueError('invalid label header')
        pairs = [(r['id'],int(r['y'])) for r in reader]
    truth = dict(pairs)
    if len(truth) != len(pairs) or set(truth) != set(rows): raise ValueError('id mismatch')
    correct = sum(rows[i] == y for i,y in truth.items())
    accuracy = correct / len(truth)
    checks = {'parameters': parameters <= PARAMETER_LIMIT, 'steps': steps <= 1000, 'epochs': epochs <= 100}
    ok = all(checks.values())
    return {'accuracy': accuracy if ok else float('inf'), 'accuracy_unguarded': accuracy,
            'correct': correct, 'n': len(truth), 'guard_ok': ok, 'guards': checks,
            'parameters': parameters, 'steps': steps, 'epochs': epochs,
            'score_rule_worst': 0, 'model_sha256': digest}

if __name__ == '__main__':
    try: result = score(*sys.argv[1:])
    except Exception as e: result = {'accuracy': float('inf'), 'guard_ok': False, 'reason': str(e), 'score_rule_worst': 0}
    print(json.dumps(result))
