"""Trusted isolated fresh fit: the same 4000 training rows in dev and formal replay."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import read_table, score_csv
from private_eval import Evaluator, remaining_time


def checked_pack(path):
    root = Path(path)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol.get('task') != 'Che-06' or protocol.get('protocol') != 'private-policy-v2':
        raise ValueError('wrong trusted evaluation pack')
    if protocol.get('evaluation') in ('development', 'formal') and protocol.get('n_train') != 4000:
        raise ValueError('validation must remain excluded from development and formal training')
    fit_ids, eval_ids = [], []
    for key, ids, count in [('train', fit_ids, protocol['n_train']), ('eval', eval_ids, protocol['n_eval'])]:
        with h5py.File(root / (key + '.h5'), 'r') as f:
            if set(f) != {'id', 'X'} or f['X'].shape != (count, 39553):
                raise ValueError('wrong descriptor population or schema')
            ids.extend(x.decode('utf-8') for x in f['id'][:])
            if len(ids) != len(set(ids)):
                raise ValueError('duplicate input ids')
    if set(fit_ids) & set(eval_ids):
        raise ValueError('fit/evaluation overlap')
    if list(read_table(root / 'train_labels.csv', 'y')) != fit_ids:
        raise ValueError('fit label alignment')
    if list(read_table(root / 'labels.csv', 'y')) != eval_ids:
        raise ValueError('evaluation label alignment')
    return protocol


def score(pred_path, labels_path):
    private = Path(labels_path)
    protocol = checked_pack(private)
    with tempfile.TemporaryDirectory(prefix='qued-private-') as tmp:
        root = Path(tmp)
        fit_bundle = root / 'fit-input'
        predict_bundle = root / 'predict-input'
        fit_bundle.mkdir()
        predict_bundle.mkdir()
        resources = root / 'empty-resources'
        resources.mkdir()
        for filename in ('train.h5', 'train_labels.csv'):
            shutil.copyfile(private / filename, fit_bundle / filename)
        shutil.copyfile(private / 'eval.h5', predict_bundle / 'eval.h5')
        (fit_bundle / 'request.json').write_text(json.dumps(dict(protocol, phase='fit')) + '\n')
        (predict_bundle / 'request.json').write_text(json.dumps(dict(protocol, phase='predict')) + '\n')
        # Fit cannot see evaluator-supplied evaluation features or labels. Its
        # inert fitted-state archive is consumed only by another fresh worker.
        with Evaluator(pred_path, 'Che-06', cpus=max(1, min(16, len(os.sched_getaffinity(0)))),
                       data_dir=resources) as evaluator:
            try:
                fit = evaluator.run(fit_bundle)
                with Evaluator(fit['output_path'], 'Che-06', cpus=max(1, min(16, len(os.sched_getaffinity(0)))),
                               data_dir=resources) as predictor:
                    replay = predictor.run(predict_bundle)
                    result = score_csv(replay['output_path'], private / 'labels.csv')
            except Exception:
                # Evaluator exceptions can carry candidate stderr. Do not return
                # that text, private paths, or any private observations.
                raise ValueError('isolated fit or prediction failed') from None
            result.update(protocol='private-policy-v2', evaluation=protocol['evaluation'],
                          authorized_training_rows=protocol['n_train'],
                          fixed_descriptor_columns=39553, private_replay_ok=True,
                          policy_elapsed_s=fit['elapsed_s'] + replay['elapsed_s'],
                          fit_elapsed_s=fit['elapsed_s'], predict_elapsed_s=replay['elapsed_s'],
                          fit_predict_separated=True,
                          training_provenance='isolated training features and labels; all-row method compliance also requires patch audit')
            remaining_time()
            return result


if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception:
        result = {'mae': float('inf'), 'rmse': float('inf'), 'coverage_ok': False,
                  'private_replay_ok': False, 'reason': 'trusted replay or output validation failed'}
    print(json.dumps(result))
