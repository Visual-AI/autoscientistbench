"""Fit a fresh regressor on the explicitly supplied training descriptor rows."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from private_eval import export_artifact

BOB_WIDTH = 39340
WIDTH = 39553


def labels(path, ids):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'y']:
            raise ValueError('training label schema')
        rows = list(reader)
    if [r['id'] for r in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError('training labels must match every training row in order')
    y = np.array([float(r['y']) for r in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite training labels')
    return y


def chunks(n, size=64):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def ids_of(f):
    return [value.decode('utf-8') for value in f['id'][:]]


def fit_model(bundle, recipe, destination):
    bundle, destination = Path(bundle), Path(destination)
    config = json.loads(Path(recipe).read_text())
    config.update(objective='reg:squarederror', tree_method='hist', device='cpu',
                  n_jobs=max(1, min(int(os.environ.get('BENCH_CPUS', '1')), len(os.sched_getaffinity(0)))),
                  random_state=int(os.environ.get('BENCH_SEED', '0')))
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(bundle / 'train.h5', 'r') as train:
        if train['X'].shape[1] != WIDTH:
            raise ValueError('expected fixed BOB+DQM descriptor width')
        train_ids = ids_of(train)
        y = labels(bundle / 'train_labels.csv', train_ids)
        n = len(train_ids)
        # The original eleven DQM StandardScalers are column-wise. One scaler
        # over their concatenation has exactly the same definition. Both this
        # stage and the complete-descriptor scaler see only active fit rows.
        dqm = StandardScaler()
        for s in chunks(n):
            dqm.partial_fit(train['X'][s, BOB_WIDTH:])
        whole = StandardScaler()
        for s in chunks(n):
            block = train['X'][s]
            block[:, BOB_WIDTH:] = dqm.transform(block[:, BOB_WIDTH:])
            whole.partial_fit(block)
        # Constant training columns cannot split a tree. This training-only
        # internal filtering retains the supplied descriptor definition and rows;
        # it can change which active columns stochastic subsampling selects.
        active = np.flatnonzero(whole.var_ > 0)
        if active.size == 0:
            active = np.array([0])
        with tempfile.TemporaryDirectory(prefix='qued-fit-') as tmp:
            x = np.memmap(Path(tmp) / 'train.f32', dtype=np.float32, mode='w+', shape=(n, len(active)))
            for s in chunks(n):
                block = train['X'][s]
                block[:, BOB_WIDTH:] = dqm.transform(block[:, BOB_WIDTH:])
                x[s] = whole.transform(block)[:, active]
            x.flush()
            model = XGBRegressor(**config)
            model.fit(x, y)
            model.save_model(destination / 'booster.ubj')
            np.savez(destination / 'preprocessing.npz', dqm_mean=dqm.mean_, dqm_scale=dqm.scale_,
                     whole_mean=whole.mean_, whole_scale=whole.scale_, active=active)
            # This trace stays in private fitted state. It is diagnostic rather
            # than a certificate that arbitrary edited code used every row.
            (destination / 'training.json').write_text(json.dumps({
                'training_rows': n,
                'dqm_scaler_rows': int(dqm.n_samples_seen_),
                'descriptor_scaler_rows': int(whole.n_samples_seen_),
                'descriptor_columns': WIDTH, 'active_training_columns': len(active),
                'n_estimators': config['n_estimators'], 'fresh_fit': True,
            }, indent=2) + '\n')
            del x


def predict_model(bundle, model_dir, output):
    bundle, model_dir, output = Path(bundle), Path(model_dir), Path(output)
    model = XGBRegressor(n_jobs=max(1, int(os.environ.get('BENCH_CPUS', '1'))))
    model.load_model(model_dir / 'booster.ubj')
    with np.load(model_dir / 'preprocessing.npz', allow_pickle=False) as data:
        state = {key: data[key] for key in data.files}
    with h5py.File(bundle / 'eval.h5', 'r') as evaluation, output.open('w', newline='') as f:
        if evaluation['X'].shape[1] != WIDTH:
            raise ValueError('expected fixed BOB+DQM descriptor width')
        eval_ids = ids_of(evaluation)
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        for s in chunks(len(eval_ids)):
            block = evaluation['X'][s]
            block[:, BOB_WIDTH:] = (block[:, BOB_WIDTH:] - state['dqm_mean']) / state['dqm_scale']
            block = ((block - state['whole_mean']) / state['whole_scale'])[:, state['active']].astype(np.float32)
            values = model.predict(block)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite prediction')
            writer.writerows(zip(eval_ids[s], map(float, values)))


def main():
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--out', required=True)
    policy = commands.add_parser('policy')
    for key in ('model', 'input', 'output'):
        policy.add_argument('--' + key, required=True)
    a = p.parse_args()
    if a.command == 'train':
        destination = Path(a.out)
        destination.mkdir(parents=True, exist_ok=True)
        config = json.loads((Path(__file__).parent / 'config.json').read_text())
        (destination / 'recipe.json').write_text(json.dumps(config, indent=2) + '\n')
    else:
        phase = json.loads((Path(a.input) / 'request.json').read_text())['phase']
        if phase == 'fit':
            with tempfile.TemporaryDirectory(prefix='qued-state-') as tmp:
                fit_model(a.input, Path(a.model) / 'recipe.json', tmp)
                export_artifact(Path(__file__).resolve().parent, tmp, a.output, 'Che-06', include_dependencies=False)
        elif phase == 'predict':
            predict_model(a.input, a.model, a.output)
        else:
            raise ValueError('unknown policy phase')


if __name__ == '__main__':
    main()
