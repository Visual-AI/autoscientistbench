import json,sys,math,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
def metrics(expected,estimated):
 # Names below the 0.5% call threshold enter only if expected, so padding cannot enlarge the denominator.
 keys=sorted(set(expected)|{k for k,v in estimated.items() if v>=.005})
 if not keys or abs(sum(expected.values())-1)>=1e-8:raise ValueError('invalid expected abundances')
 errors=[estimated.get(k,0)-expected.get(k,0) for k in keys]
 return {'rmse':math.sqrt(statistics.mean(x*x for x in errors)), 'mae':statistics.mean(abs(x) for x in errors),'expected_lineages_missed_at_0_5pct':sum(estimated.get(k,0)<0.005 for k,v in expected.items() if v>0),'non_simulated_lineage_mass':sum(v for k,v in estimated.items() if expected.get(k,0)==0),'estimated_total_mass':sum(estimated.values())}
def score(pred,labels):
 expected=labels['expected']
 if set(pred)!=set(expected):raise ValueError('missing or extra samples')
 per={key:metrics(values,pred[key]) for key,values in expected.items()};miss=max(r['expected_lineages_missed_at_0_5pct'] for r in per.values());mass=max(r['non_simulated_lineage_mass'] for r in per.values());raw=statistics.mean(r['rmse'] for r in per.values());passed=miss<=1 and mass<=.02
 return {'rmse':raw if passed else float('inf'),'unconstrained_rmse':raw,'constraints_pass':passed,'max_missed_lineages':miss,'max_spurious_mass':mass,'samples':per}
if __name__=='__main__':print(json.dumps(score(read(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text()))))
