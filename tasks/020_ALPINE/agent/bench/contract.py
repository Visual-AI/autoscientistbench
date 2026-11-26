import json,re
POOLS=['hifi_trac','hifi_trbc','ont_trac','ont_trbc']
def unique(items):
 d={}
 for key,value in items:
  if key in d:raise ValueError('duplicate JSON key')
  d[key]=value
 return d
def read(path):
 with open(path) as f:p=json.load(f,object_pairs_hook=unique)
 if not isinstance(p,dict) or set(p)!={'pools'} or not isinstance(p['pools'],dict) or set(p['pools'])!=set(POOLS):raise ValueError('four pools required')
 for pool,rows in p['pools'].items():
  if not isinstance(rows,dict) or not rows:raise ValueError('missing read predictions')
  for uid,label in rows.items():
   if not re.fullmatch('[0-9a-f]{32}',uid) or not isinstance(label,str) or not re.fullmatch('[A-Za-z0-9_/-]{1,100}',label):raise ValueError('invalid read id or class')
 return p['pools']
