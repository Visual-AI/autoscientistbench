import csv
import math
HEADER=['id']+[f'pred_{i}' for i in range(5)]
def read_predictions(path):
    rows={}
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=HEADER:raise ValueError('expected header '+','.join(HEADER))
        for row in r:
            if None in row or None in row.values():raise ValueError('ragged row')
            key=row['id']
            if not key or key in rows:raise ValueError('empty or duplicate id')
            values=[float(row[c]) for c in HEADER[1:]]
            if any(not math.isfinite(v) or v!=int(v) or not 0<=v<5 for v in values):raise ValueError('cluster IDs must be finite integers 0..4')
            rows[key]=[int(v) for v in values]
    if not rows:raise ValueError('empty predictions')
    return rows
