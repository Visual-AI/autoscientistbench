"""Export replayable preparation/solver code, excluding untrusted outer models."""
import argparse
import tempfile
from private_eval import export_artifact


def main():
    parser = argparse.ArgumentParser()
    for key in ('code', 'model', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    # Models are recreated by the same submitted train.sh under trusted timing.
    # Including the outer result would both duplicate a large runtime and allow
    # unmeasured preparation to influence scored execution.
    with tempfile.TemporaryDirectory(prefix='tensor-empty-model-') as model:
        export_artifact(args.code, model, args.output, '058_Tensor-ScaledGD')


if __name__ == '__main__':
    main()
