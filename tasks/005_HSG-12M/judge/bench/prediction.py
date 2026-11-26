"""Frozen text-table parsing shared by validation and scoring."""
import csv
import math
import re

HEADER = ['id', 'pred', 'parameters', 'steps', 'model_sha256', 'epochs']

def read_predictions(path):
    rows = {}; metadata = set()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER: raise ValueError('prediction header must be '+','.join(HEADER))
        for row in reader:
            if None in row or any(v is None for v in row.values()): raise ValueError('malformed row')
            i = row['id']
            if not i or i in rows: raise ValueError('empty or duplicate id')
            values = []
            for key in ('pred','parameters','steps','epochs'):
                value = float(row[key])
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError(key+' must be a finite nonnegative integer')
                values.append(int(value))
            if values[0] >= 24: raise ValueError('class outside 0..23')
            if not re.fullmatch('[0-9a-f]{64}',row['model_sha256']): raise ValueError('invalid model digest')
            rows[i] = values[0]
            metadata.add((*values[1:], row['model_sha256']))
    if not rows or len(metadata) != 1: raise ValueError('empty table or inconsistent model metadata')
    return rows, next(iter(metadata))
