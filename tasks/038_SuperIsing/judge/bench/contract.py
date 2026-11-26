import json,math
REQUIRED=[(6,'table6',[.0662,.0793,.0615]),(7,'table7',[.0698,.0774,.0722]),(8,'table8',[.0690,.0792,.0757]),(9,'table9',[.0680,.0809,.0776]),(9,'example9',[.0698080753241176,.0773934248387392,.07224789112627583])]
def read(path,mode=None):
 with open(path) as f:d=json.load(f)
 if not isinstance(d,dict) or set(d)!={'mode','fixed_couplings','cases'}:raise ValueError('wrong prediction fields')
 if d['mode'] not in ['full','smoke'] or mode is not None and d['mode']!=mode:raise ValueError('wrong mode')
 if d['fixed_couplings']!=[1.5,.25,1.0]:raise ValueError('fixed Hamiltonian couplings changed')
 cases=d['cases']
 if not isinstance(cases,list) or not 1<=len(cases)<=45:raise ValueError('bad case count')
 tags=set()
 for r in cases:
  if not isinstance(r,dict) or set(r)!={'nmf','tag','parameters'}:raise ValueError('candidate spectra/scores are not accepted')
  if type(r['nmf']) is not int or r['nmf'] not in ([4] if d['mode']=='smoke' else [6,7,8,9]):raise ValueError('bad N')
  if not isinstance(r['tag'],str) or not r['tag'].isalnum() or r['tag'] in tags:raise ValueError('bad tag')
  tags.add(r['tag']);p=r['parameters']
  if not isinstance(p,list) or len(p)!=3 or not all(type(x) in (int,float) and math.isfinite(x) for x in p):raise ValueError('bad parameters')
 if d['mode']=='smoke':
  if len(cases)!=1 or cases[0]['tag']!='smoke':raise ValueError('bad smoke fixture')
 else:
  if len(cases)<5:raise ValueError('required calibration cases absent')
  for r,(n,tag,p) in zip(cases,REQUIRED):
   if (r['nmf'],r['tag'],r['parameters'])!=(n,tag,p):raise ValueError('required calibration changed')
  for i,r in enumerate(cases[5:],1):
   if r['nmf']!=9 or r['tag']!=f'local{i:02d}':raise ValueError('invalid local ED schedule')
 return d
