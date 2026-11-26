"""Prepare the method; actual fitting occurs inside each isolated private fold."""
import argparse
import json
from pathlib import Path
from build_native import build

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    build(output)
    (output/'parameters.json').write_text(json.dumps({
        'kappa': 1e-5, 'depth': 4, 'lambda_': .001, 'n_thresholds': 20,
        'thresholds_strategy': 'quantile', 'verbose': False, 'min_leaf_node_size': 0}) + '\n')

if __name__ == '__main__':
    main()
