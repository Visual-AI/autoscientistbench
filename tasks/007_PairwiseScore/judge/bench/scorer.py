"""Released MMTV and capped POT Wasserstein. Binding guard failure value is 1.0.
Malformed or nonfinite tensors produce Infinity. Grouping/baselines come only from labels.
"""
import os, sys, json, warnings
from pathlib import Path
CPU_BUDGET=int(os.environ.get('BENCH_CPUS',os.environ.get('OMP_NUM_THREADS','128')))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from validate import load_prediction

def metric_job(args):
    iid,sample,target,n=args
    from paper_metrics import mmtv, wasserstein_dist
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w=float(wasserstein_dist(sample[:n],target[:n]));m=float(mmtv(sample,target))
    return {'id':iid,'wasserstein':w,'mmtv':m,'ot_iteration_warning':any('numItermax' in str(x.message) for x in caught)}

def check_samples(samples, lab):
    # Exact empirical target copies violate the observations-only protocol.
    # This catches literal label copies (including row permutations), not arbitrary
    # draws from a memorized true density; the latter still requires patch audit.
    for iid, sample in samples.items():
        target=lab[iid+'__target']
        if np.all(sample==sample[0]):
            raise ValueError('collapsed sample distribution: KDE is undefined')
        if np.array_equal(np.sort(sample,axis=0),np.sort(target,axis=0)):
            raise ValueError('exact evaluation-target marginal copy violates observations-only protocol')

def measure(pred,labels):
    with np.load(labels,allow_pickle=False) as z:lab={k:z[k] for k in z.files}
    ids=lab['ids'].tolist();shapes={i:lab[i+'__target'].shape for i in ids}
    samples=load_prediction(pred,shapes)
    check_samples(samples,lab)
    jobs=[(i,samples[i],lab[i+'__target'],int(lab[i+'__w_count'])) for i in ids]
    # Start large dense transports first to limit the final straggler under the scorer wall cap.
    jobs.sort(key=lambda job: (-job[3], -job[1].shape[1]))
    from concurrent.futures import ProcessPoolExecutor
    # Dense OT may allocate about 24 GB per large instance. Bound concurrent solves.
    with ProcessPoolExecutor(max_workers=min(12,max(1,CPU_BUDGET),len(jobs))) as pool:rows=list(pool.map(metric_job,jobs))
    return rows,lab

def aggregate(rows,lab):
    groups={}
    for r in rows:
        i=r['id'];g=int(lab[i+'__group']);groups.setdefault(g,[]).append((r,float(lab[i+'__baseline_w'])))
    summary=[]
    for g,values in sorted(groups.items()):
        w=float(np.mean([r['wasserstein'] for r,b in values]));b=float(np.mean([b for r,b in values]))
        assert np.isfinite(b) and b>0,'baseline W must be frozen and positive'
        summary.append({'group':g,'mmtv':float(np.mean([r['mmtv'] for r,b in values])),'wasserstein':w,'baseline_w':b,'w_ratio':w/b})
    raw=float(np.mean([r['mmtv'] for r in summary]));avg=float(np.mean([r['w_ratio'] for r in summary]));peak=float(max(r['w_ratio'] for r in summary))
    assert np.isfinite([raw,avg,peak]).all(),'nonfinite metrics'
    passed=avg<=1.05 and peak<=1.40
    return dict(mmtv=raw if passed else 1.0,raw_mmtv=raw,w_ratio_mean=avg,w_ratio_max=peak,guard_passed=passed,groups=summary,instances=rows)

def main(pred,labels):
    try:rows,lab=measure(pred,labels);result=aggregate(rows,lab)
    except Exception as e:result={'mmtv':float('inf'),'invalid':str(e)}
    print(json.dumps(result))
if __name__=='__main__':main(*sys.argv[1:])
