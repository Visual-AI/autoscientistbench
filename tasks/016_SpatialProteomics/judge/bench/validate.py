#!/usr/bin/env python3
"""validate.py PRED INPUT_DIRECTORY; exact fixed QC ID set, no duplicates/non-finite labels."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_io import read_table


def validate(pred, inputs):
    files = sorted(Path(inputs).glob('*.npz'))
    if not files:
        raise ValueError('Empty input directory')
    ids = set()
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            cells = data['qc_cells']
        if cells.ndim != 1 or len(cells) != len(set(cells.tolist())):
            raise ValueError('Invalid input QC cell IDs')
        ids.update(f'{file.stem}:{int(i)}' for i in cells)
    rows = read_table(pred)
    if set(rows) != ids:
        raise ValueError(f'Cell ID set mismatch: missing={len(ids-set(rows))}, extra={len(set(rows)-ids)}')
    print(f'ok: {len(ids)} QC cells')


if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            raise ValueError('usage: validate.py PRED INPUT_DIRECTORY')
        validate(*sys.argv[1:])
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
