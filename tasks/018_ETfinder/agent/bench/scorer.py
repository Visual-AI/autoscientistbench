import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
from eval_curated_overlap import compare
def score(assemblies,labels):
 if len(set(assemblies))!=len(assemblies):raise ValueError('Duplicate assembly')
 if not set(assemblies)<=labels['hosts'].keys():raise ValueError('Unknown assembly')
 pred=[labels['hosts'][a] for a in assemblies]
 if labels['population']!='all':pred=[r for r in pred if r['population']==labels['population']]
 m=compare(pred,labels['references'],'species',labels['matching'])
 valid=labels['minimum_candidates']<=len(assemblies)<=labels['maximum_candidates']
 return {'f1':m['f1'] if valid else 0.0,'unconstrained_f1':m['f1'],'constraints_pass':valid,'candidate_count_global':len(assemblies),'candidate_count_scored':len(pred),'n_reference':m['n_reference'],'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],'population':labels['population']}
if __name__=='__main__':
 labels=json.load(open(sys.argv[2]));print(json.dumps(score(read(sys.argv[1],labels['hosts']),labels)))
