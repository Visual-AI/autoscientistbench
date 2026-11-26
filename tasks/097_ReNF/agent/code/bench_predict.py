"""Standalone forecast using the saved validation-selected EMA model."""
import argparse,json,os,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from models import ReNF_alpha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 seed=int(os.environ.get('BENCH_SEED','1'));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 torch.set_num_threads(int(os.environ.get('BENCH_CPUS','8')))
 root=Path(a.model);meta=json.loads((root/'meta.json').read_text());args=SimpleNamespace(**meta)
 net=ReNF_alpha.Model(args,[args.seq_len,1,7]).float()
 state=torch.load(root/meta['checkpoint'],map_location='cpu',weights_only=True)
 # thop.profile installs accounting buffers; they are not inference parameters.
 state={k:v for k,v in state.items() if not k.endswith(('total_ops','total_params'))}
 net.load_state_dict(state);net.eval()
 with np.load(a.input,allow_pickle=False) as d:
  ids=d['id'];lengths=d['history_len'];assert (lengths>=args.seq_len).all()
  x=d['x'][:,-args.seq_len:,:].astype(np.float64)
 mean=np.array(meta['mean']);std=np.array(meta['std']);x=((x-mean)/std).astype(np.float32)
 preds=[]
 with torch.no_grad():
  for i in range(0,len(ids),args.batch_size):
   preds.append(net(torch.from_numpy(x[i:i+args.batch_size]))[-1].numpy())
 pred=np.concatenate(preds).astype(np.float64)*std+mean
 with open(a.output,'wb') as f:np.savez(f,id=ids,pred=pred,seq_len=np.array(args.seq_len),features=np.array(args.features))
if __name__=='__main__':main()
