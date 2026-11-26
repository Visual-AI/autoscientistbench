"""Public configuration/export and isolated causal policy entry points."""
import argparse
import json
import os
from pathlib import Path

TASK = '030_Conformal-Optimistic-Prediction'


def train(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).resolve().parent / 'method_config.json').read_text())
    (output / 'method.json').write_text(json.dumps(config, indent=2) + '\n')


def export(model, input_path, output):
    from private_eval import export_artifact
    request = json.loads(Path(input_path).read_text())
    if request != {'task': TASK, 'protocol': 'private-policy-v2'}:
        raise ValueError('invalid public request')
    code = Path(__file__).resolve().parent
    reference = Path(os.environ.get('BENCH_BENCH_DIR', code.parent / 'bench')) / 'reference_env.lock'
    # Reuse the frozen reference interpreter only for its exact baseline lock.
    # Changed pins are exported by the canonical runtime.
    unchanged = reference.is_file() and (code / 'env.lock').read_bytes() == reference.read_bytes()
    export_artifact(code, model, output, TASK, include_dependencies=not unchanged)


def policy(model, input_path, output):
    from online_cop import OnlineCOP
    from private_rpc import call
    if os.environ.get('BENCH_PRIVATE_EVAL') != '1':
        raise RuntimeError('causal policy requires private replay')
    request = json.loads(Path(input_path).read_text())
    config = json.loads((Path(model) / 'method.json').read_text())
    method = OnlineCOP(config['learning_rates'][request['cell']], request['burn'],
                       alpha=config['tail_alpha'], scale=config['scale'],
                       proportional_lr=config['proportional_lr'])
    while True:
        event = call('next')
        if event['op'] == 'done':
            break
        lower, upper = method.predict(event['forecast'])
        feedback = call('submit', {'t': event['t'], 'lower': lower, 'upper': upper})
        method.observe(event['forecast'], feedback['y'])
    Path(output).write_text('{"complete":true}\n')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    sub.add_parser('train').add_argument('--out', required=True)
    for name in ('predict', 'policy'):
        p = sub.add_parser(name)
        for field in ('model', 'input', 'output'):
            p.add_argument('--' + field, required=True)
    args = parser.parse_args()
    if args.stage == 'train':
        train(args.out)
    elif args.stage == 'predict':
        export(args.model, args.input, args.output)
    else:
        policy(args.model, args.input, args.output)


if __name__ == '__main__':
    main()
