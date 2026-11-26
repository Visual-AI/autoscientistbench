"""Original game/metric and counted oracle budget, with private execution."""
import csv
import json
import multiprocessing
import os
import resource
import tempfile
import time
from pathlib import Path
import numpy as np
from private_eval import Evaluator,remaining_time
from private_handler import Handler, Budget
from legacy_scorer import score
TASK_ID = 'icml-3011-AsymP-GDA'
MEMORY_BYTES = 8*1024**3
_RUN_CONTEXT = None


def _run_policy(job):
    ev,budgets,tmp,steps,affinity = _RUN_CONTEXT
    index,iid,run_id,text = job
    inp=tmp/f'{index}-{run_id}.json'
    inp.write_text(json.dumps({'id':iid,'run_id':run_id,'steps_per_run':steps}))
    handler=Handler(iid,run_id,text,budgets[iid],seed=int(ev.manifest['algorithm_seed']))
    out=ev.run(inp,handler=handler,timeout_s=None,affinity=affinity)
    trusted=handler.accepted_record()
    submitted=json.loads(Path(out['output_path']).read_text())
    if json.dumps(submitted,sort_keys=True) != json.dumps(trusted,sort_keys=True):
        raise ValueError('worker output differs from the latest trusted strategy export')
    return (iid,run_id),trusted


def evaluate(pred,labels):
    global _RUN_CONTEXT
    labels=Path(labels)
    config=json.loads((labels/'manifest.json').read_text())
    if config['task_id'] != TASK_ID or config['protocol'] != 'private-policy-v2':
        raise ValueError('wrong private game container')
    max_updates,steps=int(config['max_updates']),int(config['steps_per_run'])
    if config.get('scope') == 'full' and (max_updates != 1000000 or steps != 500000):
        raise ValueError('full evaluation budget changed')
    affinity=sorted(os.sched_getaffinity(0))[:16]
    os.sched_setaffinity(0,affinity)
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    memory_limit=min(x for x in (MEMORY_BYTES,soft,hard) if x != resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_AS,(memory_limit,hard))
    with tempfile.TemporaryDirectory(prefix='leduc-inputs-') as folder:
        tmp=Path(folder)
        infrastructure_start=time.monotonic()
        with Evaluator(pred,TASK_ID,cpus=len(affinity),timeout_s=None) as ev:
            artifact_load_s=time.monotonic()-infrastructure_start
            # Archive transport and solving are reported separately for diagnostics.
            solve_start=time.monotonic()
            remaining=remaining_time()
            deadline=solve_start+remaining if remaining is not None else None
            ev.started=solve_start
            with np.load(labels/'input.npz',allow_pickle=False) as z:
                data=dict(z)
            ids=[str(x) for x in data['id']]
            if ids != config['ids'] or len(set(ids)) != len(ids):
                raise ValueError('private id mismatch')
            jobs=[(i,iid,p,str(data['game'][i])) for i,iid in enumerate(ids) for p in (0,1)]
            n_workers=min(len(affinity),len(jobs))
            budgets={iid:Budget(max_updates,shared=n_workers>1,deadline=deadline) for iid in ids}
            _RUN_CONTEXT=(ev,budgets,tmp,steps,affinity)
            if n_workers>1:
                # Native LiteEFG graph construction has process-global state.
                # Separate trusted processes match the original independent runs.
                pool=multiprocessing.get_context('fork').Pool(n_workers)
                try:
                    pairs=pool.map_async(_run_policy,jobs).get(timeout=remaining_time())
                    pool.close()
                except BaseException:
                    pool.terminate()
                    raise
                finally:
                    pool.join()
            else:
                pairs=list(map(_run_policy,jobs))
            records=dict(pairs)
            combined=tmp/'trusted.csv'
            with combined.open('w',newline='') as f:
                w=csv.writer(f)
                w.writerow(['id','instance','player','infoset','action','probability','updates','traversals','oracle'])
                for iid in ids:
                    pair=[records[(iid,p)] for p in (0,1)]
                    traversals=sum(s['traversals'] for s in pair)
                    if traversals != budgets[iid].used:
                        raise ValueError('trusted aggregate traversal mismatch')
                    for p in (0,1):
                        for j,(_,values) in enumerate(pair[p]['players'][p]):
                            for a,value in enumerate(values):
                                w.writerow([f'{iid}:{p+1}:{j}:{a}',iid,p+1,j,a,repr(value),traversals/2,traversals,1])
            solve_wall_s=time.monotonic()-solve_start
            remaining_time()
        metrics=score(combined,labels/'metric_labels.npz')
    _RUN_CONTEXT=None
    metrics.update(trusted_oracle=True,protocol_version=2,
                   verification_scope=config.get('scope','full'),
                   solve_wall_s=solve_wall_s,
                   artifact_load_s=artifact_load_s,worker_cpus=len(affinity),
                   concurrent_runs=n_workers)
    return metrics
