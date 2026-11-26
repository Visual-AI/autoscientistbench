"""Editable fit method. Output one explicit tree in original feature/target units."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

def certificate(state, n_features):
    def walk(node):
        if node['is_leaf']:
            values = [0.] * (n_features+1)
            coefficients = np.asarray(node['coefficients']).reshape(-1)
            values[0] = float(coefficients[0])
            for column, value in zip(state['continuous_idx'], coefficients[1:]):
                values[int(column)] = float(value)
            return {'coefficients': values}
        return {'feature': int(node['feature_idx'])-1, 'threshold': float(node['threshold']),
                'left': walk(node['left']), 'right': walk(node['right'])}
    return {'schema': 1, 'tree': walk(state['root'])}

def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'input', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    model_dir = Path(args.model)
    sys.path.insert(0, str(model_dir/'native'))
    from clari_tree import CLARITree
    with np.load(Path(args.input)/'train.npz', allow_pickle=False) as arrays:
        x, y = arrays['x'], arrays['y']
    model = CLARITree(**json.loads((model_dir/'parameters.json').read_text()))
    model.fit(x, y)
    Path(args.output).write_text(json.dumps(certificate(model.export_state(), x.shape[1]), allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
