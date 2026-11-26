"""Five independent private fits, trusted tree reconstruction, equal-weight mean test R²."""
import json
import math
from pathlib import Path
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import Evaluator, remaining_time
from tree_certificate import N_FEATURES, load_certificate, predict_checked, r2

TASK_ID = 'icml-2217-CLARITree'
# Public development set: the real Seoul Bike dataset of the paper's Table 2, in its released folds.
# It shares no rows with the scored population. Only the feature count and population differ.
DEVELOPMENT = 'seoul_bike_public_development_v1'
DEVELOPMENT_POPULATION = {'features': 9, 'population': 8760, 'train_rows': 7008, 'test_rows': 1752}

def score(pred_path, labels_path):
    private = Path(labels_path)
    manifest = json.loads((private/'manifest.json').read_text())
    development = manifest.get('kind') == DEVELOPMENT
    features = DEVELOPMENT_POPULATION['features'] if development else N_FEATURES
    if manifest.get('schema') != 1 or manifest.get('task') != TASK_ID or manifest.get('folds') != 5 or manifest.get('features') != features:
        raise ValueError('private fold manifest')
    if development and any(manifest.get(key) != value for key, value in DEVELOPMENT_POPULATION.items()):
        raise ValueError('development population')
    values = []
    with tempfile.TemporaryDirectory(prefix='claritree-score-') as tmp:
        root = Path(tmp)
        resources = root/'empty-resources'
        resources.mkdir()
        with Evaluator(pred_path, TASK_ID, cpus=1, timeout_s=None, data_dir=resources) as evaluator:
            for fold in range(5):
                remaining_time()
                with np.load(private/f'fold_{fold}.npz', allow_pickle=False) as data:
                    x, y, test_x, test_y = (data[k] for k in ('train_x', 'train_y', 'test_x', 'test_y'))
                if x.shape != (len(y), features) or test_x.shape != (len(test_y), features) or not np.isfinite(y).all():
                    raise ValueError('private fold arrays')
                if development and (len(y), len(test_y)) != (DEVELOPMENT_POPULATION['train_rows'], DEVELOPMENT_POPULATION['test_rows']):
                    raise ValueError('development population')
                request = root/f'fold-{fold}'
                request.mkdir()
                np.savez(request/'train.npz', x=x, y=y)
                result = evaluator.run(request)
                certificate = load_certificate(result['output_path'])
                prediction = predict_checked(certificate, x, test_x, features)
                values.append(r2(test_y, prediction))
                remaining_time()
    result = float(np.mean(values))
    if not math.isfinite(result):
        raise ValueError('nonfinite metric')
    return {'mean_test_r2': result, 'certificate_verified': True, 'isolated_folds': 5,
            'protocol': 'private-policy-v2', 'development_only': development}

def safe_score(pred_path, labels_path):
    try:
        return score(pred_path, labels_path)
    except Exception:
        # Candidate stderr, certificates and private-data exception text are never returned.
        return {'mean_test_r2': -float('inf'), 'certificate_verified': False,
                'reason': 'private replay or tree certificate rejected'}

if __name__ == '__main__':
    print(json.dumps(safe_score(*sys.argv[1:])))
