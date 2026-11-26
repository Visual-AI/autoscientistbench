#!/usr/bin/env python3
import csv,sys
from pathlib import Path
from contract import ID,predictions
def main(pred_path,input_path):
    root=Path(input_path)
    with (root/'manifest.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','file','offset','N','side']:raise ValueError('invalid input manifest columns')
        ids=set()
        for row in reader:
            if not ID.fullmatch(row['id']) or row['id'] in ids:raise ValueError('invalid/duplicate input ID')
            ids.add(row['id'])
            if not ID.fullmatch(row['file'].removesuffix('.bin')) or not row['file'].endswith('.bin'):raise ValueError('invalid shard')
            n,side,offset=int(row['N']),int(row['side']),int(row['offset'])
            if n not in (100,150) or side not in (51,101) or offset<0 or offset%8:raise ValueError('invalid observation dimensions')
            if (root/row['file']).stat().st_size<offset+n*side*side*8:raise ValueError('truncated observations')
    if not ids or set(predictions(pred_path))!=ids:raise ValueError('prediction/input ID mismatch')
    print(f'ok: {len(ids)} rows')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as exc:sys.exit(str(exc))
