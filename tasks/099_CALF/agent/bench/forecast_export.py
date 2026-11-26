"""Export a model/policy before any private test observations are released."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from private_eval import export_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads((HERE / 'forecast_config.json').read_text())
    descriptor = json.loads(Path(args.input).read_text())
    if descriptor.get('protocol') != 'private-forecast-v2' or descriptor.get('task_id') != config['task_id']:
        raise ValueError('expected this task\'s private forecast descriptor')
    resources = None
    if config['pretrained_resource']:
        resources = {'pretrained': Path(os.environ['BENCH_DATA_DIR']) / 'pretrained'}
    code = Path(os.environ.get('BENCH_CODE_DIR', os.getcwd())).resolve()
    export_artifact(code, Path(args.model), Path(args.output), config['task_id'],
                    resources=resources, include_dependencies=True)


if __name__ == '__main__':
    main()
