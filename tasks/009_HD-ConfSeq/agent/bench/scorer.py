"""Table 3 score rule, with independently evaluated certificates and Fig-4 coverage."""
import json,math,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from validate import load,check
from geometry import VolumeCalculator,compute_ellips_bbx_volume
from predicate import membership,volume as predicate_volume


def coverage(p,lab,iid):
 X=lab[iid];mu=lab[iid+'__true_mean'];key=lambda k:p[iid+'__'+k]
 if str(key('coverage_type'))=='predicate':
  return np.array([membership(cert,mu[None,:],lo,hi)[0]
                   for cert,lo,hi in zip(key('predicate'),key('lower'),key('upper'))])
 if str(key('coverage_type'))=='box_lp':
  v=np.sum(np.abs((mu-key('center'))/key('radii'))**key('power')[:,None],axis=1)
  return (v<=1)&np.all((mu>=key('lower'))&(mu<=key('upper')),axis=1)
 lam=key('lam');c=key('c');theta=key('theta');w=key('weights')
 plus=1+np.minimum(lam,c/(mu+1e-12))*(X[:-1]-mu)
 minus=1-np.minimum(lam,c/(1-mu+1e-12))*(X[:-1]-mu)
 with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
  logs_plus=np.cumsum(np.where(plus>0,np.log(plus),-np.inf),axis=0)
  logs_minus=np.cumsum(np.where(minus>0,np.log(minus),-np.inf),axis=0)
  logs=np.maximum(np.log(theta)+logs_plus,np.log(1-theta)+logs_minus)
  return np.sum(w*np.exp(logs[np.arange(49,1000,50)-1]),axis=1)<20


def score_records(rows,covs):
 """Pure aggregation, also used for an independent archived-result closure."""
 paper={r['scenario']:r for r in rows if r['role']=='paper'}
 assert set(paper)=={'effect','no_effect'},'one primary pair required'
 assert sum(r['role']=='paper' for r in rows)==2,'unique primary pair'
 raw=(paper['effect']['stop']+paper['no_effect']['stop'])/2
 cover={k:float(np.mean(np.all(np.asarray(v,dtype=bool),axis=1))) for k,v in covs.items()}
 assert set(cover)=={'bernoulli_mix','circle'}
 false={s:sum(r['false'] for r in rows if r['role']=='replicate' and r['scenario']==s) for s in paper}
 for s in paper:assert sum(r['role']=='replicate' and r['scenario']==s for r in rows)==20,'20 independent replications required'
 gates={'effect_decision':paper['effect']['decision']==1 and not paper['effect']['false'],
        'no_effect_decision':paper['no_effect']['decision']==2,
        'coverage':all(v>=.919 for v in cover.values()),'false_flags':all(v<=3 for v in false.values())}
 ok=all(gates.values())
 return dict(mean_stop_t=float(raw if ok else 500),raw_mean_stop_t=float(raw),guards_ok=ok,guards=gates,coverage=cover,false_flags=false,primary_stops={s:r['stop'] for s,r in paper.items()})


def score(p,lab):
 obs={'id':lab['id'],**{str(i):lab[str(i)] for i in lab['id']}}
 check(p,obs)
 rows=[];covs={}
 for iid in map(str,lab['id']):
  key=lambda k:p[iid+'__'+k]
  scenario=str(lab[iid+'__scenario']);role=str(lab[iid+'__role'])
  if str(lab[iid+'__kind'])=='coverage':
   covs.setdefault(scenario,[]).append(coverage(p,lab,iid));continue
  low=key('lower');up=key('upper');vol=key('volume')
  typ=str(p.get(iid+'__geometry_type','box_lp'))
  if typ=='box_lp':
   q=key('center');r=key('radii')
   expected_low=np.maximum.reduce([q-r,key('box_lower'),np.zeros_like(q)])
   expected_up=np.minimum.reduce([q+r,key('box_upper'),np.ones_like(q)])
   # Match the released numerical fallback when the coordinate projections cross.
   bad=expected_low>expected_up;expected_low[bad]=0;expected_up[bad]=1
   assert np.allclose(low,expected_low,rtol=0,atol=1e-10) and np.allclose(up,expected_up,rtol=0,atol=1e-10),'bounds do not match certificate'
  flags=(low>.5)|(up<.5);sig=flags.any(axis=1);small=vol**.1<=.02
  triggered=np.flatnonzero(sig|small)
  t=int(triggered[0])+1 if len(triggered) else 500
  dec=1 if sig[t-1] else (2 if small[t-1] else 0)
  assert int(key('stop'))==t and int(key('decision'))==dec,'reported decision/stop mismatch'
  assert np.array_equal(key('above'),low[t-1]>.5) and np.array_equal(key('below'),up[t-1]<.5),'reported flags mismatch'
  # At the selected stopping time, independently re-evaluate the complete set's
  # volume using the released 5 x 25000 sample rule and submitted RNG seed.
  if typ=='predicate':
   volume,_=predicate_volume(key('predicate')[-1],low[-1],up[-1],int(key('volume_seed')))
  else:
   power=int(key('power')[-1]);Q=np.diag(r[-1]**power)
   h=dict(calculate_volume=True,cut_to_simplex=False,interval_types=[],volume_workers=1,volume_mc_seed=int(key('volume_seed')))
   volume,_=compute_ellips_bbx_volume(h,10,q=q[-1],Q=Q,Q_inv=np.linalg.inv(Q),radii=r[-1],bbx_low=np.maximum(low[-1],key('box_lower')[-1]),bbx_high=np.minimum(up[-1],key('box_upper')[-1]),pow=power)
  assert abs(volume-vol[-1])<=1e-9*max(abs(volume),1e-100),'volume does not match certificate'
  if dec==2:
   # A volume decision needs evidence: at least 100 of the 125,000 Monte Carlo points in the set.
   proposal=VolumeCalculator.safe_positive_product(up[-1]-low[-1])
   if typ!='predicate':
    # The certificate samples from the smaller of its box and its Lp ellipsoid.
    box=VolumeCalculator.safe_positive_product(np.minimum(up[-1],key('box_upper')[-1])-np.maximum(low[-1],key('box_lower')[-1]))
    D=len(r[-1]);proposal=min(box,math.exp(D*math.log(2*math.gamma(1+1/power))-math.log(math.gamma(1+D/power))+float(np.sum(np.log(r[-1])))))
   assert proposal>0 and round(125000*float(vol[-1])/proposal)>=100,'volume decided from too few Monte Carlo points'
  false=bool(np.any((key('above').astype(bool)|key('below').astype(bool))&~lab[iid+'__true_effect']))
  rows.append(dict(scenario=scenario,role=role,stop=t,decision=dec,false=false))
 return score_records(rows,covs)

if __name__=='__main__':
 try:result=score(load(sys.argv[1]),load(sys.argv[2]))
 except Exception as e:result={'mean_stop_t':float('inf'),'protocol_ok':False,'reason':str(e)}
 print(json.dumps(result))
