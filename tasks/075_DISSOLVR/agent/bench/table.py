"""Frozen table parsing shared by validator and scorer."""
import csv,math
PRED_COLUMNS=['id']+[f'pred_{i}' for i in range(5)]+['train_s','train_only']
def read(path,columns):
 with open(path,newline='') as f:
  r=csv.DictReader(f)
  if r.fieldnames!=columns:raise ValueError(f'expected header {columns}, got {r.fieldnames}')
  rows=list(r)
 if not rows:raise ValueError('empty table')
 ids=[]
 for row in rows:
  if None in row or any(v is None for v in row.values()):raise ValueError('wrong column count')
  s=row['id']
  if not s.isdecimal() or str(int(s))!=s:raise ValueError('id must be a canonical nonnegative integer')
  ids.append(int(s))
  for c in columns:
   if c not in ('id','SMILES') and not math.isfinite(float(row[c])):raise ValueError('nonfinite '+c)
 if len(set(ids))!=len(ids):raise ValueError('duplicate ids')
 if ids!=sorted(ids):raise ValueError('rows must be sorted by numeric id')
 return rows

def prediction(path):
 rows=read(path,PRED_COLUMNS)
 for key in ['train_s','train_only']:
  if len({float(r[key]) for r in rows})!=1:raise ValueError('inconsistent '+key)
 if float(rows[0]['train_s'])<0:raise ValueError('negative training time')
 if float(rows[0]['train_only']) not in (0,1):raise ValueError('train_only must be 0 or 1')
 return rows
