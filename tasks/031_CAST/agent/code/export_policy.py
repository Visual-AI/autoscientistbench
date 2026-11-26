import argparse
import json
from pathlib import Path
from private_eval import export_artifact

def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'input', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    if request.get('task') != 'Fin-04' or request.get('protocol') != 'private-policy-v2':
        raise ValueError('invalid episode descriptor')
    export_artifact(Path(__file__).resolve().parent, args.model, args.output, 'Fin-04')

if __name__ == '__main__':
    main()
