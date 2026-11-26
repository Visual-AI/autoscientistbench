import json,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read,POOLS
from original_metric import evaluate_pool
BASE={'hifi_trac':(98.45,96.08),'hifi_trbc':(99.44,95.64),'ont_trac':(99.62,87.16),'ont_trbc':(99.63,86.91)}
def score(pred,labels):
 if set(pred)!=set(POOLS) or set(labels['pools'])!=set(POOLS):raise ValueError('four pools required')
 metrics={};guards={}
 for pool in POOLS:
  truth=labels['pools'][pool]['truth'];rows=labels['pools'][pool]['rows']
  if set(pred[pool])!=set(truth):raise ValueError('read ids differ')
  pairs=collections.Counter((group,pred[pool][uid]) for uid,group in truth.items())
  metrics[pool]=evaluate_pool(pool,rows,pairs,len(truth))
  r=metrics[pool];base=BASE[pool]
  guards[pool]={'integration':r['integration']['f1']>=round(base[0]-.2,2),'total':r['total']['f1']>=round(base[1]-.5,2)}
 valid=all(all(g.values()) for g in guards.values());raw=sum(metrics[k]['total']['f1'] for k in ['ont_trac','ont_trbc'])/2
 return {'f1':raw if valid else 0.,'unconstrained_f1':raw,'constraints_pass':valid,'guards':guards,'pools':metrics}
if __name__=='__main__':print(json.dumps(score(read(sys.argv[1]),json.load(open(sys.argv[2]))),allow_nan=False))
