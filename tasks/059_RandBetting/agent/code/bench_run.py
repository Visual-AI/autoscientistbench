"""Prepare a global betting method and export it before private observation replay."""
import argparse
import json
import os
import sys
from pathlib import Path

BENCH = Path(os.environ['BENCH_BENCH_DIR']).resolve()
sys.path.insert(0, str(BENCH))
from betting_interface import DEFAULT
from private_betting import REQUEST_PROTOCOL, check_method
from private_eval import export_artifact


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--out', required=True)
    predict = sub.add_parser('predict')
    for key in ('model', 'input', 'output'):
        predict.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # Exact original baseline, executed by the frozen arithmetic fast path.
        method = {'version': 2, 'execution': 'certificate', 'policy': dict(DEFAULT)}
        (out / 'betting.json').write_text(json.dumps(method) + '\n')
        print('Prepared global betting method')
        return
    request = json.loads(Path(args.input).read_text())
    if request != {'protocol': REQUEST_PROTOCOL}:
        raise ValueError('invalid private betting request')
    check_method(json.loads((Path(args.model) / 'betting.json').read_text()))
    export_artifact(Path(__file__).resolve().parent, args.model, args.output, 'Sta-34')
    print('Exported betting method')


if __name__ == '__main__':
    main()
