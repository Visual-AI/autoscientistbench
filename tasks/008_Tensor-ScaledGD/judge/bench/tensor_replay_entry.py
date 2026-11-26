"""Frozen worker entry for timed preparation and observation-only solving."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for key in ('model', 'input', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if os.environ.get('TENSOR_REPLAY_STAGE') == 'prepare':
        model = Path(args.output).parent / 'replayed-model'
        subprocess.run(['bash', '/ws/code/train.sh', '--out', str(model)], check=True)
        Path(args.output).write_text(json.dumps({'prepared': True}) + '\n')
    else:
        os.execv('/bin/bash', ['bash', '/ws/code/.tensor-submitted-policy.sh',
                              '--model', args.model, '--input', args.input,
                              '--output', args.output])


if __name__ == '__main__':
    main()
