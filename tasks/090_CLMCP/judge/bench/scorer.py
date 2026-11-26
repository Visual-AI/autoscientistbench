#!/usr/bin/env python3
"""Private causal execution followed by the unchanged cumulative regret metric."""
import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import SimpleQueue
sys.path.insert(0,str(Path(__file__).resolve().parent))
from private_eval import Evaluator,remaining_time
from stream_env import read_input
from replay import Replay
from metric import score as metric
TASK_ID='icml-5412-CLMCP'

def cpu_groups(package, available):
    """Allocate episode parallelism within the unchanged eight-CPU allowance."""
    if not available:
        raise ValueError('no CPU allocation')
    config=Path(package)/'code/policy_resources.json'
    requested=1
    if config.is_file():
        with config.open('rb') as stream:
            payload=stream.read(4097)
        if len(payload)>4096:
            raise ValueError('policy resource configuration is too large')
        settings=json.loads(payload)
        if not isinstance(settings,dict) or set(settings)!={'episode_cpus'}:
            raise ValueError('policy resources must contain only episode_cpus')
        requested=settings['episode_cpus']
        if type(requested) is not int or not 1<=requested<=8:
            raise ValueError('episode_cpus must be an integer from 1 to 8')
    count=min(requested,len(available))
    return [available[i:i+count] for i in range(0,len(available)-count+1,count)]

def score(pred_path, labels_path):
    input_started=time.monotonic()
    private=Path(labels_path)
    groups=read_input(private/'input.csv')
    replays=[Replay(group) for group in groups]
    input_elapsed=time.monotonic()-input_started
    with tempfile.TemporaryDirectory(prefix='clmcp-score-') as tmp:
        root=Path(tmp)
        requests=[]
        for i,group in enumerate(groups):
            p=root/f'request-{i}.json'
            p.write_text(json.dumps({'task':TASK_ID,'protocol':'private-policy-v2','horizon':len(group['x'])}))
            requests.append(p)
        with Evaluator(pred_path,TASK_ID,cpus=8,timeout_s=None) as ev:
            allocations=cpu_groups(ev.package,sorted(os.sched_getaffinity(0))[:8])
            workers=min(len(allocations),len(groups))
            available=SimpleQueue()
            for allocation in allocations[:workers]:
                available.put(allocation)
            artifact_elapsed=time.monotonic()-ev.started
            prediction_started=time.monotonic()-input_elapsed
            ev.started=prediction_started
            def run(i):
                allocation=available.get()
                try:
                    result=ev.run(requests[i],handler=replays[i],affinity=allocation)
                    if result['returncode'] != 0: raise ValueError('isolated policy failed')
                    return result['elapsed_s']
                finally:
                    available.put(allocation)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                elapsed=list(pool.map(run,range(len(groups))))
        predictions=root/'pred.csv'
        with predictions.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','pred'])
            for replay in replays: w.writerows(replay.rows())
        prediction_elapsed=time.monotonic()-prediction_started
        remaining_time()
        result=metric(predictions,private/'labels.csv')
        result.update(causality_guard_ok=True,protocol='private-policy-v2',policy_elapsed_s=sum(elapsed),
                      prediction_elapsed_s=prediction_elapsed,artifact_prepare_s=artifact_elapsed,worker_count=workers,
                      cpus_per_worker=len(allocations[0]))
        return result

if __name__=='__main__':
    try: result=score(*sys.argv[1:])
    except Exception as exc: result={'cumulative_regret':float('inf'),'valid':False,'causality_guard_ok':False,'reason':str(exc)}
    print(json.dumps(result))
