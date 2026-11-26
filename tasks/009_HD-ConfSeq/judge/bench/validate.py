"""Strict tensor prediction contract; no pickle loading."""
import sys
import numpy as np
from predicate import Predicate

def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def check(p,x):
 assert x['id'].ndim==p['id'].ndim==1,'id vector shape'
 ids=list(map(str,x['id']));pi=list(map(str,p['id']))
 assert len(pi)==len(set(pi)) and set(pi)==set(ids),'id set/duplicates'
 expected={'id'}
 for iid in ids:
  X=x[iid];D=X.shape[1]
  def get(k,shape=None):
   key=iid+'__'+k;expected.add(key);v=p[key]
   if shape is not None:assert v.shape==shape,(iid,k,'shape',v.shape,shape)
   if v.dtype.kind in 'biuf':assert np.isfinite(v).all(),(iid,k,'nonfinite')
   return v
  if D==10:
   t=int(get('stop',()));assert 1<=t<=min(500,len(X)),'stop range'
   assert float(get('stop',()))==t,'integer stop'
   for k in ['lower','upper']:get(k,(t,D))
   lo=p[iid+'__lower'];hi=p[iid+'__upper']
   assert np.all((0<=lo)&(lo<=hi)&(hi<=1)),'invalid bounds'
   typ=str(get('geometry_type',())) if iid+'__geometry_type' in p else 'box_lp'
   if typ=='box_lp':
    for k in ['center','radii','box_lower','box_upper']:get(k,(t,D))
    assert np.all(p[iid+'__radii']>0),'positive radii required'
    power=get('power',(t,));assert np.all((power>=1)&(power<=64)&(power==np.floor(power))),'power'
   elif typ=='predicate':
    certs=get('predicate',(t,));assert certs.dtype.kind=='U','JSON string certificates required'
    for cert in certs:Predicate(cert,D)
   else:raise ValueError('unknown geometry certificate')
   vol=get('volume',(t,));assert np.all((vol>=0)&(vol<=1)),'volume'
   dec=get('decision',());assert dec in [0,1,2],'decision'
   for k in ['above','below']:
    flags=get(k,(D,));assert np.all((flags==0)|(flags==1)),'flags'
   vs=get('volume_seed',());assert 0<=vs<2**32 and vs==np.floor(vs),'volume seed'
  elif D==5:
   typ=str(get('coverage_type',()))
   if typ=='hedged':
    lam=get('lam',(len(X)-1,D));assert np.all(lam>=0),'negative lambda'
    c=get('c',(D,));theta=get('theta',(D,));w=get('weights',(D,))
    assert np.all((c>0)&(c<1)) and np.all((theta>0)&(theta<1)),'bet parameters'
    assert np.all(w>0) and abs(w.sum()-1)<1e-10,'portfolio weights'
   elif typ=='box_lp':
    for k in ['lower','upper','center','radii']:get(k,(20,D))
    low=p[iid+'__lower'];high=p[iid+'__upper']
    assert np.all((0<=low)&(low<=high)&(high<=1)),'coverage bounds'
    assert np.all(p[iid+'__radii']>0),'coverage radii'
    power=get('power',(20,));assert np.all((power>=1)&(power<=64)),'coverage power'
   elif typ=='predicate':
    for k in ['lower','upper']:get(k,(20,D))
    low=p[iid+'__lower'];high=p[iid+'__upper']
    assert np.all((0<=low)&(low<=high)&(high<=1)),'coverage bounds'
    certs=get('predicate',(20,));assert certs.dtype.kind=='U','JSON string certificates required'
    for cert in certs:Predicate(cert,D)
   else:raise ValueError('unknown coverage certificate')
  else:raise ValueError('unexpected dimension')
 assert set(p)==expected,'unexpected/missing prediction keys'
 return True
if __name__=='__main__':
 try:check(load(sys.argv[1]),load(sys.argv[2]))
 except Exception as e:sys.exit('invalid prediction: '+str(e))
 print('valid')
