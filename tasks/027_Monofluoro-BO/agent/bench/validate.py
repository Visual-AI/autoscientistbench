#!/usr/bin/env python3
"""Strict CSV contract; standard library only. validate.py PRED INPUT."""
import csv
import math
import sys


def read_prediction(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'pred_cls', 'pred']:
            raise ValueError('Expected header id,pred_cls,pred')
        result = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed row')
            i = row['id']
            if not i or i in result:
                raise ValueError('Empty or duplicate id')
            if row['pred_cls'] not in ('0', '1'):
                raise ValueError('pred_cls must be 0 or 1')
            p = float(row['pred'])
            if not math.isfinite(p):
                raise ValueError('Non-finite yield prediction')
            result[i] = (int(row['pred_cls']), p)
    if not result:
        raise ValueError('Empty predictions')
    return result


def main(pred, inp):
    values = read_prediction(pred)
    with open(inp, newline='') as f:
        ids = [r['id'] for r in csv.DictReader(f)]
    if len(ids) != len(set(ids)) or set(ids) != set(values):
        raise ValueError('IDs must exactly match input, without duplicates')
    print(f'ok: {len(values)} rows')


if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception as exc:
        sys.exit(str(exc))
