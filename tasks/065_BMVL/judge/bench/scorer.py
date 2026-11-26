"""Trusted private fitting and classification, with the original BMVL metric."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from private_eval import Evaluator, remaining_time
from protocol import TASK_ID, DEVELOPMENT, CLEAN_FLOOR, load_plan, load_dataset, corrupt, run_fold, aggregate


def score(pred_path,labels_path):
    plan=load_plan(labels_path)
    accuracies={}
    elapsed=0.0
    budget=max(1,min(32,int(os.environ.get('BENCH_CPUS',os.environ.get('OMP_NUM_THREADS','32')))))
    cores=sorted(os.sched_getaffinity(0))[:budget]
    threads=min(4,len(cores))
    workers=max(1,len(cores)//threads)
    groups=Queue()
    for i in range(workers):groups.put(cores[i*threads:(i+1)*threads])
    # Fresh private runtime state for every fit and every inference condition.
    with Evaluator(pred_path,TASK_ID,cpus=len(cores),timeout_s=None) as evaluator:
        for name in plan['datasets']:
            arrays,views,y=load_dataset(Path(labels_path).parent,plan,name)
            classes=plan['files'][name]['num_classes']
            def evaluate_seed(seed):
                affinity=groups.get()
                try:
                    remaining_time()
                    train,test=arrays[f'train_{seed}'],arrays[f'test_{seed}']
                    original_test=[x[test] for x in views]
                    observed={eta:corrupt(original_test,seed,eta/100) for eta in (0,50)}
                    predicted,seconds=run_fold(evaluator,[x[train] for x in views],y[train],observed,
                                               classes,seed,affinity=affinity)
                    accuracy={eta:float(np.mean(predicted[eta]==y[test])*100.0) for eta in (0,50)}
                    return seed,accuracy,seconds
                finally:groups.put(affinity)
            with ThreadPoolExecutor(max_workers=min(workers,len(plan['seeds']))) as pool:
                for seed,accuracy,seconds in pool.map(evaluate_seed,plan['seeds']):
                    elapsed+=seconds
                    for eta in (0,50):accuracies[name,seed,eta]=accuracy[eta]
    result=aggregate(accuracies,plan['datasets'],plan['seeds'],CLEAN_FLOOR[plan['kind']])
    result.update(protocol='private-policy-v2',isolation_guard_ok=True,policy_elapsed_s=elapsed,
                  development=plan['kind']==DEVELOPMENT,worker_pool_size=workers)
    return result


def safe_score(pred_path,labels_path):
    try:return score(pred_path,labels_path)
    except Exception:
        # Runtime exceptions can contain candidate stderr and private data.
        # Never return those strings, paths, traces, arrays or worker logs.
        return {'acc_eta50':float('-inf'),'isolation_guard_ok':False,'reason':'private_evaluation_failed'}

if __name__=='__main__':print(json.dumps(safe_score(*sys.argv[1:])))
