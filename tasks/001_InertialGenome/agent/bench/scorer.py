"""Authors' test_model distance metrics, recomputed from padded coordinates."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr
import torch
from validate import read_prediction

THRESHOLD={'test':.095,'val':.12}

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

def score(pred_path,labels_path):
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','8')))
    with np.load(labels_path,allow_pickle=False) as lab:
        ids=lab['ids'].tolist();xyz=read_prediction(pred_path,ids);rows=[]
        for i,ident in enumerate(ids):
            # Keep the authors' padding-dependent denominator, including diagonal.
            coords=torch.from_numpy(xyz[i:i+1])
            dist=torch.cdist(coords,coords)
            dist=dist/torch.amax(dist,dim=(1,2),keepdim=True)
            contacts=torch.tensor(lab[f'contact_{i}'],dtype=torch.float32)
            n=int(lab[f'n_{i}']);assert contacts.shape==(n,n)
            valid=(contacts>1e-8)&(~torch.eye(n,dtype=torch.bool))
            truth=torch.zeros_like(contacts)
            truth[valid]=(1.0/contacts[valid])**.5
            truth[valid]=truth[valid]/truth[valid].max()
            target=truth[valid].numpy();estimate=dist[0,:n,:n][valid].numpy()
            dscc=float(spearmanr(target,estimate)[0])
            drmse=float(np.sqrt(np.mean((estimate-target)**2)))
            if not np.isfinite([dscc,drmse]).all():raise ValueError('undefined distance metric')
            rows.append({'id':ident,'dscc':dscc,'drmse':drmse})
    dscc=float(np.mean([r['dscc'] for r in rows]));drmse=float(np.mean([r['drmse'] for r in rows]))
    threshold=THRESHOLD[split_of(labels_path)]
    passes=drmse<=threshold
    return {'dscc':dscc if passes else float('inf'),'mean_dscc':dscc,'mean_drmse':drmse,
        'drmse_threshold':threshold,'guard_pass':passes,'scientific_score':dscc if passes else -1,
        'per_group':rows}

def main(pred,labels):
    try:result=score(pred,labels)
    except Exception as e:result={'dscc':float('inf'),'scientific_score':-1,'guard_pass':False,'error':str(e)}
    print(json.dumps(result,allow_nan=True))
if __name__=='__main__':main(*sys.argv[1:])
