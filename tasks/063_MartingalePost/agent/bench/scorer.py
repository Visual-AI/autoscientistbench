import collections,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read_predictions

def score(elapsed,rows,labels):
 truth=labels['observations']
 if set(rows)!=set(truth):raise ValueError('IDs mismatch')
 counts=collections.Counter(t['n'] for t in truth.values())
 if set(counts)!={20,100,500} or any(v!=labels['replications_per_n'] for v in counts.values()):raise ValueError('incomplete grid')
 cells=collections.defaultdict(list)
 for key,t in truth.items():
  pred=rows[key]
  for j,theta in enumerate(t['theta']):
   lo,hi=pred['lo'][j],pred['hi'][j]
   val=hi-lo+40*max(lo-theta,0)+40*max(theta-hi,0)
   cells[(t['n'],j)].append(math.sqrt(t['n'])*val)
 means={str(n)+'_'+['mu1','s1','s12'][j]:statistics.mean(v) for (n,j),v in cells.items()}
 value=statistics.mean(means.values())
 return {'interval_score':value,'unconstrained_interval_score':value,'elapsed_s':elapsed,'resource_timing_requires_patch_audit':True,'truth_access_requires_patch_audit':True,'cells':means,'instances':len(rows)}
if __name__=='__main__':
 elapsed,rows=read_predictions(sys.argv[1]);print(json.dumps(score(elapsed,rows,json.load(open(sys.argv[2])))))
