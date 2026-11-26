"""Fresh isolated causal replay for each cell, then the fixed Table-2 metric."""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import CELLS, aggregate, cell_metrics, load_replay
from private_eval import Evaluator, remaining_time
from replay import Replay

TASK = '030_Conformal-Optimistic-Prediction'


def _score(pred_path, labels_path):
    cells, synthetic = load_replay(Path(labels_path) / 'labels.npz')
    groups = {}
    with tempfile.TemporaryDirectory(prefix='cop-replay-') as tmp:
        # Each run has fresh writable state. Package, model and frozen utilities
        # are read-only, and no other cell's files or process are mounted.
        with Evaluator(pred_path, TASK, cpus=8) as evaluator:
            for index, cell in enumerate(CELLS):
                data = cells[cell]
                replay = Replay(data['forecast'], data['y'])
                request = Path(tmp) / ('cell-' + str(index) + '.json')
                request.write_text(json.dumps({'task': TASK, 'protocol': 'private-policy-v2',
                    'cell': cell, 'n': len(data['y']), 'burn': data['burn']}))
                evaluator.run(request, handler=replay)
                lower, upper = replay.intervals()
                groups[cell] = cell_metrics(data['y'], lower, upper, data['burn'])
                remaining_time()
    result = aggregate(groups)
    result.update(protocol='private-policy-v2', causality_guard_ok=True,
                  independent_cell_state=True, synthetic_development=synthetic)
    return result


def score(pred_path, labels_path):
    try:
        return _score(pred_path, labels_path)
    except Exception:
        # Candidate stdout/stderr, private outcomes and exception contents never
        # cross this boundary, including Evaluator failures with raw stderr.
        return {'geomean_width': math.inf, 'all_constraints_pass': False,
                'causality_guard_ok': False, 'reason': 'private replay rejected'}


if __name__ == '__main__':
    print(json.dumps(score(*sys.argv[1:]), allow_nan=True))
