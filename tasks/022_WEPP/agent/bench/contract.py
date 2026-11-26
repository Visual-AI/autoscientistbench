import json,math
def no_duplicates(pairs):
 d={}
 for key,value in pairs:
  if key in d:raise ValueError('duplicate JSON key')
  d[key]=value
 return d
def read(path):
 with open(path) as f:d=json.load(f,object_pairs_hook=no_duplicates)
 if not isinstance(d,dict) or set(d)!={'samples'} or not isinstance(d['samples'],list):raise ValueError('wrong prediction structure')
 result={}
 for s in d['samples']:
  if not isinstance(s,dict) or set(s)!={'id','abundances'} or not isinstance(s['id'],str) or s['id'] in result:raise ValueError('bad sample identity')
  if not isinstance(s['abundances'],list):raise ValueError('abundances must be pairs')
  values={}
  for item in s['abundances']:
   if not isinstance(item,list) or len(item)!=2:raise ValueError('bad abundance pair')
   name,value=item
   if not isinstance(name,str) or not name or name in values:raise ValueError('duplicate or empty lineage')
   if type(value) not in [int,float] or not math.isfinite(value) or value<0:raise ValueError('invalid abundance')
   values[name]=float(value)
  result[s['id']]=values
 return result
