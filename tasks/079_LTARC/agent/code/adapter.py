"""Public preparation/export and isolated baseline fit/certification entry points."""
import argparse
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

TASK = 'icml-3290-LTARC'
DEFAULT_CONFIG = {'lambda_range': [.01, .8], 'lambda_n': 100, 'lookahead': [0, 1]}


def train(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'method.json').write_text(json.dumps(DEFAULT_CONFIG, indent=2) + '\n')


def export(model, input_path, output):
    from private_eval import export_artifact
    request = json.loads(Path(input_path).read_text())
    if request.get('task') != TASK or request.get('protocol') != 'private-policy-v2':
        raise ValueError('invalid task descriptor')
    # The trusted scorer uses its frozen reference environment. Reuse identical
    # baseline pins; export the locked distributions when method dependencies change.
    code = Path(__file__).resolve().parent
    reference = Path(os.environ.get('BENCH_BENCH_DIR', code.parent / 'bench')) / 'reference_env.lock'
    def pins(path):
        return sorted(line.strip() for line in path.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith('#'))
    changed_dependencies = pins(code / 'env.lock') != pins(reference)
    export_artifact(Path(__file__).resolve().parent, model, output, TASK,
                    include_dependencies=changed_dependencies)


def policy(model, input_path, output):
    import numpy as np
    from certificate import frame, decisions
    from method.fft import run_fft
    from method.weights_p_bound import WeightsDecision
    if os.environ.get('BENCH_PRIVATE_EVAL') != '1':
        raise ValueError('policy is a private replay entry point')
    logging.disable(logging.CRITICAL)
    input_path = Path(input_path)
    request = json.loads((input_path / 'request.json').read_text())
    with np.load(input_path / 'data.npz', allow_pickle=False) as arrays:
        x, a, y = arrays['x'], arrays['a'], arrays['y']
    names, boolean = request['names'], request['boolean']
    df = frame(x, names, boolean, a, y)
    def propensity(features, action):
        return np.full(len(features), .5)
    weighter = WeightsDecision(df, names, 1, propensity, .01)
    if request['phase'] == 'fit':
        settings = json.loads((Path(model) / 'method.json').read_text())
        args = SimpleNamespace(**settings, n_bins=40, n_splits=3)
        fft, policies = run_fft(df, args, names, weighter)
        candidates = []
        for tree in policies:
            candidates.append({'rules': [{'criterion': int(key), 'decision': int(decision)}
                                          for key, decision in tree[:-1]],
                               'default': int(tree[-1])})
        document = {'candidates': candidates}
    elif request['phase'] == 'certify':
        # Same released Hoeffding-Bentkus selection. No tree can be fitted here.
        best, selected = 1000., None
        for index, tree in enumerate(request['candidates']):
            d = decisions(tree, x, request['criteria'])
            objective, _, _ = weighter.get_obj_and_constr(d)
            bound = weighter.get_hoeffding(d)
            if not tree['rules'] and d[0] == 0:
                bound = 0
            if bound < request['tau'] and objective < best:
                best, selected = objective, index
        if selected is None:
            raise ValueError('no candidate certified')
        document = {'selected': selected}
    else:
        raise ValueError('invalid private phase')
    Path(output).write_text(json.dumps(document) + '\n')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--out', required=True)
    for name in ('predict', 'policy'):
        p = sub.add_parser(name)
        for field in ('model', 'input', 'output'):
            p.add_argument('--' + field, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.out)
    elif args.command == 'predict':
        export(args.model, args.input, args.output)
    else:
        policy(args.model, args.input, args.output)


if __name__ == '__main__':
    main()
