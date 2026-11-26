"""Benchmark protocol entry points for the official staged XGBoost baseline."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
from reaction_discovery.prediction.staged_model_cli import (
    build_model_bundle, featurize_dataframe, load_model_family,
    run_staged_prediction, save_model_family,
)


def train(args):
    data = Path(args.data) / 'train'
    df = pd.read_csv(data / 'input.csv').merge(pd.read_csv(data / 'labels.csv'), on='id', validate='one_to_one')
    df = df.rename(columns={'y': 'yield'})
    X, y, used = featurize_dataframe(df, bits=2048, radius=2, require_yield=True)
    if len(used) != len(df):
        raise ValueError('Invalid training rows')
    seed = int(os.environ.get('BENCH_SEED', args.seed))
    bundle = build_model_bundle('xgboost', random_state=seed)
    # Identical fit calls and productive-only mask to run_staged_prediction.
    bundle['classifier'].fit(X, (y > 0).astype(int))
    bundle['regressor'].fit(X[y > 0], y[y > 0])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_model_family(out, 'xgboost', bundle)
    (out / 'metadata.json').write_text(json.dumps({'seed': seed, 'bits': 2048, 'radius': 2, 'n_train': len(y)}))
    print(f'Trained staged XGBoost on {len(y)} reactions')


def predict(args):
    df = pd.read_csv(args.input)
    X, _, used = featurize_dataframe(df, bits=2048, radius=2, require_yield=False)
    if len(used) != len(df):
        raise ValueError('Invalid prediction rows')
    bundle = load_model_family(Path(args.model), 'xgboost')
    for key in ('classifier', 'regressor'):
        bundle[key].set_params(n_jobs=int(os.environ.get('BENCH_CPUS', '8')))
    cls, pred = run_staged_prediction(**bundle, X_train=None, X_test=X)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    # Convert float32 to Python floats to preserve every bit in the text output.
    pd.DataFrame({'id': used.id, 'pred_cls': cls, 'pred': pred.astype(float)}).to_csv(out, index=False)
    print(f'Wrote {len(used)} predictions')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='command', required=True)
    t = sp.add_parser('train'); t.add_argument('--data', required=True); t.add_argument('--seed', type=int, default=1); t.add_argument('--out', required=True)
    p = sp.add_parser('predict'); p.add_argument('--model', required=True); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    args = ap.parse_args(); train(args) if args.command == 'train' else predict(args)
