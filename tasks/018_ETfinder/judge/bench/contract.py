import json,re
def read(path,allowed=None):
 with open(path) as f:d=json.load(f)
 if not isinstance(d,dict) or set(d)!={'assemblies'}:raise ValueError('Expected only assemblies')
 rows=d['assemblies']
 if not isinstance(rows,list) or len(rows)>50000:raise ValueError('Invalid assembly list')
 if any(not isinstance(x,str) or not re.fullmatch(r'GCF_\d+\.\d+',x) for x in rows):raise ValueError('Invalid assembly identifier')
 if len(set(rows))!=len(rows):raise ValueError('Duplicate assembly identifier')
 if allowed is not None and not set(rows)<=set(allowed):raise ValueError('Unknown assembly')
 return rows
