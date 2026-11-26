"""Frozen BMVL research population, corruption and fold-local replay."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np

TASK_ID = 'cvpr-427-BMVL'
DATASETS = ('Caltech-5V', '100Leaves', 'handwritten', 'LandUse_21', 'Scene15')
POPULATION = {'Caltech-5V':(1400,7,5), '100Leaves':(1600,100,3),
              'handwritten':(2000,10,6), 'LandUse_21':(2100,21,3), 'Scene15':(4485,15,3)}
SEEDS = tuple(range(10))
FORMAL = 'bmvl-original-v1'
DEVELOPMENT = 'ccv-public-development-v1'
# A Table 2 dataset of the paper: real data that shares no rows with the five scored datasets.
DEVELOPMENT_POPULATION = {'CCV':(6773,20,3)}
DEVELOPMENT_SEEDS = (0,1,2,3,4)
CLEAN_FLOOR = {FORMAL:91.80,DEVELOPMENT:58.00}


def digest_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): result.update(block)
    return result.hexdigest()


def load_plan(path):
    path=Path(path)
    plan=json.loads(path.read_text())
    if plan['task_id'] != TASK_ID: raise ValueError('task mismatch')
    if plan['kind'] == FORMAL:
        if plan['datasets'] != list(DATASETS) or plan['seeds'] != list(SEEDS):
            raise ValueError('all five datasets and ten original partitions required')
    elif plan['kind'] == DEVELOPMENT:
        if plan['datasets'] != list(DEVELOPMENT_POPULATION) or plan['seeds'] != list(DEVELOPMENT_SEEDS):
            raise ValueError('invalid development population')
    else: raise ValueError('unknown evaluation population')
    if set(plan['files']) != set(plan['datasets']): raise ValueError('missing dataset')
    for item in plan['files'].values():
        if Path(item['name']).name != item['name']: raise ValueError('invalid data path')
    return plan


def load_dataset(root, plan, name):
    item=plan['files'][name]
    path=Path(root)/item['name']
    if digest_file(path) != item['sha256']: raise ValueError('data digest mismatch')
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    y=arrays['y']
    if y.dtype.kind not in 'iu' or y.ndim != 1: raise ValueError('invalid labels')
    views=[arrays[f'x{i}'] for i in range(item['num_views'])]
    n=len(y)
    classes=item['num_classes']
    if (n,classes,len(views)) != {**POPULATION,**DEVELOPMENT_POPULATION}[name]:
        raise ValueError('original population required')
    if sorted(np.unique(y).tolist()) != list(range(classes)): raise ValueError('class population mismatch')
    if any(x.ndim!=2 or len(x)!=n or x.dtype!=np.float32 or not np.isfinite(x).all() for x in views):
        raise ValueError('invalid views')
    for seed in plan['seeds']:
        train,test=arrays[f'train_{seed}'],arrays[f'test_{seed}']
        if train.dtype.kind not in 'iu' or test.dtype.kind not in 'iu': raise ValueError('invalid split')
        if not np.array_equal(np.sort(np.concatenate([train,test])),np.arange(n)):
            raise ValueError('split must partition complete population')
        if len(test)!=int(np.ceil(n*.2)):
            raise ValueError('original split size required')
    return arrays,views,y


def corrupt(views,seed,ratio):
    """Exactly the original feature corruption; never emits clean indicators."""
    rng=np.random.default_rng(seed)
    n,m=len(views[0]),len(views)
    output=[np.copy(x) for x in views]
    count=int(ratio*n)
    if count==0:return output
    selected=rng.choice(n,size=count,replace=False)
    indicators=np.ones((n,m),dtype=np.int64)
    all_views=np.arange(m)
    keep=int(np.ceil(m/2.0))
    for row in selected:
        retained=rng.choice(all_views,size=keep,replace=False)
        indicators[row,np.setdiff1d(all_views,retained)]=0
    for view in range(m):
        rows=np.where(indicators[:,view]==0)[0]
        values=views[view][rows]
        order=rng.permutation(values.shape[0])
        output[view][rows]=values[order]
    return output


def read_predictions(path,n,classes):
    path=Path(path)
    if path.stat().st_size>max(4096,n*32): raise ValueError('oversized prediction')
    obj=json.loads(path.read_text())
    if not isinstance(obj,dict) or set(obj)!={'pred'}: raise ValueError('invalid prediction keys')
    pred=obj['pred']
    if not isinstance(pred,list) or len(pred)!=n: raise ValueError('prediction count mismatch')
    if any(type(x) is not int or x<0 or x>=classes for x in pred): raise ValueError('invalid class prediction')
    return np.asarray(pred,dtype=np.int64)


def run_fold(evaluator,train_views,train_y,test_views,num_classes,seed,affinity=None):
    """One fit; opaque private state copied separately to each inference worker.

    test_views is already corrupted by the trusted caller for each condition.
    No original IDs, clean correspondence, test targets or other-fold arrays are
    mounted. The runtime gives every call fresh temporary and output directories.
    """
    with tempfile.TemporaryDirectory(prefix='bmvl-fold-') as temporary:
        root=Path(temporary)
        fit=root/'fit';fit.mkdir()
        request={'stage':'fit','num_views':len(train_views),'num_classes':int(num_classes),'fit_seed':int(seed)}
        (fit/'request.json').write_text(json.dumps(request))
        np.savez(fit/'train.npz',y=train_y,**{f'x{i}':x for i,x in enumerate(train_views)})
        result=evaluator.run(fit,affinity=affinity)
        state=Path(result['output_path'])
        predictions={}
        elapsed=result['elapsed_s']
        for condition in (0,50):
            infer=root/f'infer-{condition}';infer.mkdir()
            # Condition identity and clean/noisy correspondence are not inputs.
            (infer/'request.json').write_text(json.dumps({'stage':'predict','num_views':len(train_views),
                                                         'num_classes':int(num_classes)}))
            np.savez(infer/'input.npz',**{f'x{i}':x for i,x in enumerate(test_views[condition])})
            shutil.copyfile(state,infer/'fit.out')
            reply=evaluator.run(infer,affinity=affinity)
            predictions[condition]=read_predictions(reply['output_path'],len(test_views[condition][0]),num_classes)
            elapsed+=reply['elapsed_s']
        return predictions,elapsed


def aggregate(accuracies,datasets=DATASETS,seeds=SEEDS,clean_floor=CLEAN_FLOOR[FORMAL]):
    required={(name,seed,eta) for name in datasets for seed in seeds for eta in (0,50)}
    if set(accuracies)!=required: raise ValueError('incomplete evaluation population')
    if any(not np.isfinite(x) or not 0<=x<=100 for x in accuracies.values()):
        raise ValueError('invalid accuracy')
    # The source rounds each seed percentage before either level of averaging.
    # All datasets have the same fixed number of seeds. Integer hundredths avoid
    # a floating accumulation artifact turning the exact 91.80 boundary into a
    # spurious guard failure, while retaining the source's per-seed np.round.
    totals={eta:sum(int(np.rint(float(np.round(accuracies[name,s,eta],2))*100))
                    for name in datasets for s in seeds) for eta in (0,50)}
    denominator=100*len(datasets)*len(seeds)
    clean=totals[0]/denominator
    noisy=totals[50]/denominator
    return {'acc_eta50':noisy if clean>=clean_floor else 0.0,'acc_eta50_raw':noisy,'acc_eta0':clean,
            'clean_guard_ok':clean>=clean_floor,'datasets':len(datasets),'folds':len(datasets)*len(seeds)}
