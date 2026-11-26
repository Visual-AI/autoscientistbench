import json,math

def read_predictions(path):
 obj=json.load(open(path));elapsed=float(obj['elapsed_s'])
 if not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid elapsed time')
 rows={}
 for row in obj['predictions']:
  key=row['id']
  if not isinstance(key,str) or not key or key in rows:raise ValueError('duplicate or invalid id')
  lo=[float(x) for x in row['lo']];hi=[float(x) for x in row['hi']]
  if len(lo)!=3 or len(hi)!=3 or not all(math.isfinite(x) for x in lo+hi):raise ValueError('invalid endpoints')
  if any(a>b for a,b in zip(lo,hi)):raise ValueError('reversed interval')
  rows[key]={'lo':lo,'hi':hi}
 if not rows:raise ValueError('empty predictions')
 return elapsed,rows
