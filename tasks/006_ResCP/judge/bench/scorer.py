"""Private per-cell replay, followed by the original 12-cell guarded metric."""
import csv,json,os,sys,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from private_eval import Evaluator,remaining_time
from replay import Replay
from metric import score as metric
TASK_ID='044_reservoir-conformal-prediction'
def score(pred_path,labels_path):
    input_started=time.monotonic()
    private=Path(labels_path)
    with np.load(private/'input.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    cells=sorted(k[:-4] for k in arrays if k.endswith('__id'))
    if len(cells)!=12:raise ValueError('expected all 12 original cells')
    replays=[Replay(arrays,c) for c in cells]
    input_elapsed=time.monotonic()-input_started
    with tempfile.TemporaryDirectory(prefix='rescp-score-') as tmp:
        root=Path(tmp);requests=[]
        for i,cell in enumerate(cells):
            p=root/f'request-{i}.json';p.write_text(json.dumps({'task':TASK_ID,'protocol':'private-policy-v2','cell':cell}));requests.append(p)
        with Evaluator(pred_path,TASK_ID,cpus=128,timeout_s=None) as ev:
            artifact_elapsed=time.monotonic()-ev.started
            prediction_started=time.monotonic()-input_elapsed
            ev.started=prediction_started
            available=min(128,len(os.sched_getaffinity(0)))
            threads=min(4,available)
            workers=min(12,max(1,available//threads))
            def run(i):
                result=ev.run(requests[i],handler=replays[i])
                if result['returncode']!=0:raise ValueError('isolated policy failed')
                return result['elapsed_s']
            with ThreadPoolExecutor(max_workers=workers) as pool:elapsed=list(pool.map(run,range(12)))
        pred=root/'pred.csv'
        with pred.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','lower','upper'])
            for replay in replays:w.writerows(replay.rows())
        prediction_elapsed=time.monotonic()-prediction_started
        remaining_time()
        result=metric(pred,private/'labels.npz');result.update(causality_guard_ok=True,protocol='private-policy-v2',policy_elapsed_s=sum(elapsed))
        result.update(prediction_elapsed_s=prediction_elapsed,artifact_prepare_s=artifact_elapsed,worker_pool_size=workers)
        return result
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as exc:result={'normalized_winkler':float('inf'),'causality_guard_ok':False,'reason':str(exc)}
    print(json.dumps(result))
