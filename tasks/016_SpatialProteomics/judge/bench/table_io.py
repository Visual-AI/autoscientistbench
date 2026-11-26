"""Frozen strict CSV parsing shared by validation and scoring."""
import csv
import re

CLASSES = ['B cell', 'T cell', 'Myeloid cell', 'Dendritic cell', 'Macrophage',
           'Stromal cell', 'Endothelial cell']
ID = re.compile(r'[0-9a-f]{20}:[1-9][0-9]*\Z')


def read_table(path):
    result = {}
    with open(path, newline='', encoding='utf-8') as f:
        rows = csv.reader(f)
        if next(rows, None) != ['id', 'label']:
            raise ValueError('Header must be id,label')
        for row in rows:
            if len(row) != 2:
                raise ValueError('Every row needs exactly two fields')
            sid, label = row
            if not ID.fullmatch(sid):
                raise ValueError('Malformed cell ID')
            if sid in result:
                raise ValueError('Duplicate cell ID')
            if label.strip().lower() in {'nan', 'inf', '-inf', '+inf', 'infinity', '-infinity'}:
                raise ValueError('Non-finite label')
            # Empty labels and arbitrary out-of-class values mean mismatch, not omission.
            result[sid] = int(label) if re.fullmatch(r'-?[0-9]+', label) else -1
    return result
