"""Protocol adapter for the released deterministic threshold + argmax pipeline."""
import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import time
import numpy as np
import xarray as xr
import spatialproteomics  # Registers released accessors.
from spatialproteomics.constants import Features, Layers, Props, Labels

MARKER_CT_DICT = {
    'PAX5': 'B cell', 'CD3': 'T cell', 'CD11b': 'Myeloid cell',
    'CD11c': 'Dendritic cell', 'CD68': 'Macrophage', 'CD90': 'Stromal cell',
    'Podoplanin': 'Stromal cell', 'CD31': 'Endothelial cell', 'CD34': 'Endothelial cell',
}
DEFAULTS = {'PAX5': .8, 'CD3': .7, 'CD11b': .8, 'CD11c': .9,
            'CD68': .9, 'CD31': .95, 'CD34': .95, 'CD90': .95, 'Podoplanin': .95}
CLASSES = ['B cell', 'T cell', 'Myeloid cell', 'Dendritic cell', 'Macrophage',
           'Stromal cell', 'Endothelial cell']


def process_core(path):
    start = time.monotonic()
    with np.load(path, allow_pickle=False) as inp:
        ds = xr.Dataset({
            '_image': (('channels', 'y', 'x'), inp['image'].astype(np.float32)),
            '_segmentation': (('y', 'x'), inp['segmentation']),
        }, coords={'channels': inp['channels'], 'y': inp['y'], 'x': inp['x']})
        expected = inp['qc_cells']
    # Same calls, ordering, dtype and defaults as the reproduction.
    ds = ds.pp.add_observations(properties=('label', 'centroid', 'area'))
    ds = ds.pp.remove_outlying_cells()
    channels = list(MARKER_CT_DICT)
    ds = ds.pp.threshold(quantile=[DEFAULTS[c] for c in channels], channels=channels)
    ds = ds.pp.add_quantification(func='intensity_mean')
    ds = ds.pp.transform_expression_matrix(method='arcsinh')
    ds = ds.la.predict_cell_types_argmax(MARKER_CT_DICT)
    np.testing.assert_array_equal(ds.cells.values, expected)
    names = ds.la._label_to_dict(Props.NAME)
    lab_ids = ds[Layers.OBS].sel(features=Features.LABELS).values.astype(int)
    labels = [names.get(i, Labels.UNLABELED) for i in lab_ids]
    rows = [(f'{path.stem}:{int(cell)}', CLASSES.index(label) if label in CLASSES else -1)
            for cell, label in zip(ds.cells.values, labels)]
    print(f'{path.stem}: {len(rows)} cells, {time.monotonic()-start:.2f}s, '
          f'worker peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.3f} GiB', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    if not (Path(a.model) / 'NO_TRAINING').is_file():
        raise ValueError('Missing model marker; call train.sh first')
    seed = int(os.environ.get('BENCH_SEED', '1'))
    random.seed(seed)
    np.random.seed(seed)
    files = sorted(Path(a.input).glob('*.npz'))
    if not files:
        raise ValueError('No input instances')
    # Four independent cores, as in the reproduction, bounded by the CPU allowance.
    workers = min(4, int(os.environ.get('BENCH_CPUS', '1')), len(files))
    rows = []
    with mp.get_context('spawn').Pool(workers, maxtasksperchild=1) as pool:
        for batch in pool.imap(process_core, files):
            rows.extend(batch)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(sorted(rows))
    tmp.replace(out)


if __name__ == '__main__':
    main()
