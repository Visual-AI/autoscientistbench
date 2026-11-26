#!/usr/bin/env python3
"""Validate exact CSV schema, finite estimates, and ordered id equality."""
import csv
import math
import sys


def read_predictions(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'pred']:
            raise ValueError('header must be id,pred')
        rows = list(reader)
    ids, values = [], []
    for row in rows:
        if set(row) != {'id','pred'} or not row['id']:
            raise ValueError('malformed row')
        value = float(row['pred'])
        if not math.isfinite(value):
            raise ValueError('nonfinite prediction')
        ids.append(row['id']); values.append(value)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate predictions')
    return ids, values


def main(pred, inp):
    ids, values = read_predictions(pred)
    with open(inp, newline='') as f:
        wanted = [row['id'] for row in csv.DictReader(f)]
    if len(wanted) != len(set(wanted)) or ids != wanted:
        raise ValueError('ids or row order differ from input')
    print('ok:', len(ids), 'rows')

if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception as e:
        sys.exit(str(e))
