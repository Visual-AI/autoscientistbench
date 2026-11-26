import json
def read(path):
 with open(path) as f:d=json.load(f)
 if not isinstance(d,dict) or set(d)!={'samples'} or not isinstance(d['samples'],dict) or not d['samples']:raise ValueError('bad sample map')
 for name,rows in d['samples'].items():
  if not isinstance(rows,dict) or not rows:raise ValueError('bad read map')
  for rid,r in rows.items():
   if not isinstance(rid,str) or not rid or not isinstance(r,dict) or set(r)!={'status','start'}:raise ValueError('bad read schema')
   if r['status'] not in ['retained','no_start','filtered']:raise ValueError('bad status')
   if r['status']=='retained':
    if type(r['start']) is not int or r['start']<0:raise ValueError('bad start')
   elif r['start'] is not None:raise ValueError('removed read must have null start')
 return d['samples']
def check(rows,sequences):
 if set(rows)!=set(sequences):raise ValueError('read population changed')
 for rid,r in rows.items():
  if r['status']=='retained' and r['start']>=len(sequences[rid]):raise ValueError('start outside read')
