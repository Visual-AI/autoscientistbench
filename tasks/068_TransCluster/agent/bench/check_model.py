"""Replay probe. Run inside the same isolated candidate stage as predict.
Trusted rollout recomputes all predictions. Counts enforce the fixed capacity budget; timings are diagnostic.
Input provenance and independent-instance behavior require patch audit.
"""
import argparse,json,os,sys,tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rollout import run,BASELINE_PARAMETERS
from scorer import objective

def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--code',required=True,type=Path);p.add_argument('--input',required=True,type=Path);p.add_argument('--pred',required=True,type=Path);p.add_argument('--baseline-timing',type=Path);p.add_argument('--seed',type=int,default=1);p.add_argument('--cpus',type=int,default=128);p.add_argument('--report',type=Path);p.add_argument('--timing-input',type=Path)
    a=p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        replay=Path(td)/'replay.out';run(a.code,a.model,a.input,replay,a.seed,a.cpus)
        with np.load(replay) as z:r={k:z[k] for k in z.files}
        with np.load(a.pred) as z:q={k:z[k] for k in z.files}
        if a.timing_input:
            reps=[]
            for rep in range(3):
                timing_out=Path(td)/'timing.out';run(a.code,a.model,a.timing_input,timing_out,a.seed,a.cpus)
                with np.load(timing_out) as z:reps.append(z['inference_s'].copy())
            inference_s=np.mean(reps,axis=0)
        else:inference_s=r['inference_s']
    torch.set_num_threads(min(4,a.cpus))
    with np.load(a.input) as z:x=z['x']
    def log_values(centers):
        return torch.stack([torch.cat([objective(torch.from_numpy(x[j:j+32]),torch.from_numpy(centers[m,j:j+32,-1])).log() for j in range(0,len(x),32)]) for m in range(3)])
    actual_logs,reported_logs=log_values(r['centers']),log_values(q['centers'])
    error=float((actual_logs-reported_logs).abs().max())
    same=np.array_equal(r['id'],q['id']) and bool(torch.isfinite(actual_logs).all()) and bool(torch.isfinite(reported_logs).all()) and error<=1e-4
    actual=float(inference_s.sum())
    count_ok=np.array_equal(r['parameters'],q['parameters']) and bool((r['parameters']<=BASELINE_PARAMETERS).all())
    ok=same and count_ok
    report={'ok':bool(ok),'normal_log_objective':None if ok else float('inf'),'paper_score':None if ok else 6.343,'replay_equal':bool(same),'parameter_budget_ok':bool(count_ok),'ids_equal':bool(np.array_equal(r['id'],q['id'])),'max_abs_error':float(np.max(np.abs(r['centers']-q['centers']))),'max_per_task_log_error':error,'objective_tolerance':1e-4,'parameters':r['parameters'].tolist(),'inference_s':actual}
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
