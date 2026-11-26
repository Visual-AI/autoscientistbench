"""Private R null-stream replay with authenticated draw counts."""
import contextlib,csv,hashlib,json,math,os,subprocess,sys,tempfile
from pathlib import Path
from common import FIELDS,INPUT_FIELDS,inputs,labels,predictions,rows
from private_eval import Evaluator,validate_artifact
from legacy_metric import score as numeric_score
METRIC='mean_permutations_per_hypothesis_anytime_valid_BC'
HERE=Path(__file__).resolve().parent
class Sampler:
 def __init__(self,root,trial,M,B,log):
  self.M,self.B,self.counts=M,B,[0]*M
  self.log=open(log,'w')
  self.process=subprocess.Popen([str(Path(os.environ['R_HOME'])/'bin/Rscript'),'--vanilla',str(HERE/'sampler_daemon.R'),
    str(root/'sampler_states.rds'),trial],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
  if self.process.stdout.readline().rstrip('\n')!='ready':self.close();raise RuntimeError('private R sampler failed to initialize')
 def close(self):
  if self.process.poll() is None:self.process.terminate()
  with contextlib.suppress(subprocess.TimeoutExpired):self.process.wait(timeout=3)
  if self.process.poll() is None:self.process.kill();self.process.wait()
  self.log.close()
 def __call__(self,method,args,kwargs):
  if method!='sample' or kwargs or len(args)!=2:raise ValueError('unsupported oracle operation')
  indices,end=args
  if not isinstance(indices,list) or not indices or len(indices)>self.M or any(type(i) is not int or i<1 or i>self.M for i in indices) or len(set(indices))!=len(indices):raise ValueError('invalid hypotheses')
  if type(end) is not int or not 1<=end<=self.B:raise ValueError('invalid prefix budget')
  old=[self.counts[i-1] for i in indices]
  # sampler.R batches rows without changing any prefix or random draw.
  expected=sum(max(0,end-n) for n in old)
  if expected>16384:raise ValueError('sample response requires row batching')
  self.process.stdin.write(','.join(map(str,indices))+'\t'+str(end)+'\n');self.process.stdin.flush()
  line=self.process.stdout.readline().rstrip('\n');parts=line.split('\t')
  if len(parts)!=2 or list(map(int,parts[0].split(',')))!=old:raise RuntimeError('private R sampler response invalid')
  values=list(map(float,parts[1].split(','))) if parts[1] else []
  if len(values)!=expected or not all(math.isfinite(x) for x in values):raise RuntimeError('private R sample values invalid')
  for i in indices:self.counts[i-1]=max(self.counts[i-1],end)
  return {'old':old,'values':values}
def score(pred,root):
 root=Path(root);validate_artifact(pred,'Sta-33');y,groups,_=labels(root/'labels.json')
 if inputs(root/'input.csv')!=groups:raise ValueError('private observation/protocol mismatch')
 if hashlib.sha256((root/'input.csv').read_bytes()).hexdigest()!=y['input_sha256']:raise ValueError('private observation hash mismatch')
 observed=rows(root/'input.csv',INPUT_FIELDS)
 by_trial={trial:[] for trial in groups}
 for row in observed:by_trial[row['trial_id']].append(row)
 with tempfile.TemporaryDirectory(prefix='mc-score-') as tmp:
  tmp=Path(tmp);combined=tmp/'predictions.csv';constant='f'*32;queries=0
  cpus=min(32,max(1,int(os.environ.get('BENCH_CPUS','32'))),len(os.sched_getaffinity(0)))
  with combined.open('w',newline='') as dest,Evaluator(pred,'Sta-33',cpus=cpus,timeout_s=None) as ev:
   writer=csv.DictWriter(dest,fieldnames=FIELDS);writer.writeheader()
   for trial,g in groups.items():
    inp=tmp/'trial.csv'
    with inp.open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=INPUT_FIELDS);w.writeheader()
     # Preserve the original decimal strings of the legally observed statistics.
     for row in by_trial[trial]:w.writerow({**row,'trial_id':constant})
    sampler=Sampler(root,trial,g['M'],g['B'],tmp/'sampler.log')
    try:
     run=ev.run(inp,handler=sampler);out=Path(run['output_path'])
     if out.stat().st_size>g['M']*1000:raise ValueError('oversized decision output')
     result=predictions(out,{constant:g})[constant]
     for h,(rejected,draws,declared) in result.items():
      if draws!=sampler.counts[h-1] or declared!=1:raise ValueError('reported draws differ from actual private disclosures')
      writer.writerow(dict(trial_id=trial,hypothesis=h,rejected=rejected,draws=draws,sampler_only=1))
     queries+=sum(sampler.counts)
    finally:sampler.close()
  metric=numeric_score(combined,root/'labels.json')
  # Disclosures/counts are enforced; the original method restrictions need review.
  metric['patch_audit_required']=['sampler-only decision computation','exclusion of analytic p-values, extra null simulation and truth lookup']
  return {**metric,'protocol':'private-null-values-v3','isolation_ok':True,'actual_disclosures':queries,'draw_counts_verified':True,'method_compliance_verified':False}
if __name__=='__main__':
 try:out=score(*sys.argv[1:])
 except Exception as e:out={METRIC:10000,'invalid':True,'isolation_ok':False,'reason':str(e)}
 print(json.dumps(out,allow_nan=False))
