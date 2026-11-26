import sys,json,statistics,regex
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read,check
# Retained reads with shorter tails count only up to the baseline's number of such reads in that sample.
SHORT_TAIL=1000
def score(pred,labels):
 if set(pred)!=set(labels['samples']):raise ValueError('sample population changed')
 samples={}
 for name,l in labels['samples'].items():
  rows=pred[name];check(rows,l['reads']);wt=[];purity=[];length=[];counts={s:0 for s in ['retained','no_start','filtered']}
  for rid,r in rows.items():
   counts[r['status']]+=1
   if r['status']!='retained':continue
   seq=l['reads'][rid][r['start']:];a=seq.count('GGTTAG')*6/len(seq)*100
   b=len(list(regex.finditer(r'(GGTTAG){s<=1}',seq)))*6/len(seq)*100-a
   wt.append(a);purity.append(a+b);length.append(len(seq))
  n=len(rows);mean=statistics.fmean(purity) if purity else None;threshold=l['purity_baseline']-.5;median=statistics.median(length) if length else None
  counted=len(purity)-max(0,sum(x<SHORT_TAIL for x in length)-l['short_tail_baseline'])
  samples[name]={'subtelo_pass':n,'telomeric_reads':len(purity),'identification_rate':counted/n,'mean_wt_composition':statistics.fmean(wt) if wt else None,'mean_wt_plus_1N_composition':mean,'median_vrr_telo_length':median,'constraints':{'denominator_unchanged':n==l['expected_denominator'],'purity_satisfied':mean is not None and mean>=threshold,'tail_length_kept':median is not None and median>=.9*l['median_tail_baseline']},'purity_threshold_percent':threshold,'audit':counts}
 valid=all(all(s['constraints'].values()) for s in samples.values());raw=statistics.fmean(s['identification_rate'] for s in samples.values())
 return {'identification_rate':raw if valid else 0.,'unconstrained_identification_rate':raw,'constraints_pass':valid,'samples':samples}
if __name__=='__main__':print(json.dumps(score(read(sys.argv[1]),json.load(open(sys.argv[2]))),allow_nan=False))
