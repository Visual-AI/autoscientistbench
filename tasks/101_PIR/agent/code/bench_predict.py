"""Raw-scale, independent-window inference using the authors' PIR forward."""
import argparse, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from exp.exp_long_term_forecasting_pir import Exp_Long_Term_Forecast_PIR

def calendar(starts,length):
    dates=pd.DatetimeIndex((pd.to_datetime(starts).values[:,None]+np.arange(length)[None,:]*np.timedelta64(1,'h')).reshape(-1))
    return np.stack([dates.month/12-0.5,dates.day/31-0.5,dates.dayofweek/6-0.5,dates.hour/23-0.5,dates.minute/59-0.5],axis=-1).reshape(len(starts),length,5).astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--horizon',type=int,required=True);a=p.parse_args()
    seed=int(os.environ.get('BENCH_SEED','1'));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','4')))
    h=a.horizon;out={}
    sys.path.insert(0,os.environ['BENCH_BENCH_DIR'])
    from check_model import check_horizon
    check_horizon(a.model,h)
    with np.load(a.input,allow_pickle=False) as inp:
        mask=inp['horizon']==h;ids=inp['id'][mask];raw=inp['x'][mask];starts=inp['start'][mask]
    if len(ids)==0:
        with open(a.output,'wb') as f:np.savez(f,**out)
        return
    assert raw.shape==(len(ids),96,7)
    folder=next((Path(a.model)/f'h{h}').glob('*_PIR_*'))
    args=SimpleNamespace(**json.loads((folder/'config.json').read_text()))
    args.root_path=str(folder.parent/'history');args.use_gpu=False
    scaler=json.loads((folder/'scaler.json').read_text());mean=np.array(scaler['mean']);std=np.array(scaler['std'])
    exp=Exp_Long_Term_Forecast_PIR(args)
    data,loader=exp._get_data('retrieval');exp.prepare_retrieval_index(data,loader)
    exp.model.load_state_dict(torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=True));exp.model.eval()
    x=((raw-mean)/std).astype(np.float32);marks=calendar(starts,96+h);pred=[]
    with torch.no_grad():
        for i in range(0,len(ids),32):
            bx=torch.from_numpy(x[i:i+32]);bm=torch.from_numpy(marks[i:i+32])
            y=exp.model(bx,bm[:,:96],torch.zeros((len(bx),h,7)),bm[:,96:],mode='refine')[0]
            pred.append(y.numpy().astype(np.float64)*std+mean)
    out={f'id_{h}':ids,f'pred_{h}':np.concatenate(pred)}
    with open(a.output,'wb') as f:np.savez(f,**out)
if __name__=='__main__':main()
