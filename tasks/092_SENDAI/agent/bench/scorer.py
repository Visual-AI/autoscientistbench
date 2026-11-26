#!/usr/bin/env python3
"""Mean per-run frame SSIM, with the mean of three run RMSEs <= 0.130."""
import json,sys
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction

def score(pred_path,labels_path):
    p=read_prediction(pred_path)
    with np.load(labels_path,allow_pickle=False) as lab:
        assert set(lab.files)=={'ids','targets'},'label keys'
        ids=lab['ids'];truth=lab['targets']
    assert len(ids)==len(set(ids.tolist())),'duplicate label ids'
    assert set(p['ids'].tolist())==set(ids.tolist()),'label id mismatch'
    order={v:i for i,v in enumerate(p['ids'].tolist())}
    pred=p['pred'][:,[order[v] for v in ids.tolist()]]
    assert truth.shape==(len(ids),64,64) and np.isfinite(truth).all(),'invalid truth'
    # Author outputs/evaluation use float32 arrays, including float32 square/mean.
    # Preserve submitted numeric dtype, as the original metric implementation does.
    rmse=[float(np.sqrt(np.mean((run-truth)**2))) for run in pred]
    ssim=[float(np.mean([structural_similarity(t,v,data_range=max(t.max()-t.min(),1e-8)) for t,v in zip(truth,run)])) for run in pred]
    assert np.isfinite(rmse).all() and np.isfinite(ssim).all(), 'nonfinite computed metric'
    mean_rmse=float(np.mean(rmse));mean_ssim=float(np.mean(ssim))
    budget=len(p['sensor_locs'])<=64 and int(p['lags'])<=5
    passed=mean_rmse<=.130 and budget
    return {'ssim':mean_ssim if passed else float('inf'),'raw_ssim':mean_ssim,'rmse':mean_rmse,'per_run_ssim':ssim,'per_run_rmse':rmse,'sensors':len(p['sensor_locs']),'lags':int(p['lags']),'guard_pass':bool(passed),'rule_worst':-1 if not passed else None}

if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result={'ssim':float('inf'),'guard_pass':False,'rule_worst':-1,'error':str(e)}
    print(json.dumps(result))
