"""Window-local inference from saved validation-selected checkpoints."""
import argparse,json,os,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from models.CALF import Model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    seed=int(os.environ.get('BENCH_SEED','1'));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(16,int(os.environ.get('BENCH_CPUS','1'))));torch.set_num_interop_threads(1)
    inp=np.load(a.input,allow_pickle=False);out={}
    for h in sorted(set(inp['horizon'].tolist())):
        started=time.monotonic()
        print(f'Loading horizon {h}', flush=True)
        folder=list((Path(a.model)/f'h{h}').glob('*/meta.json'))
        if len(folder)!=1:raise ValueError(f'Expected one checkpoint metadata for horizon {h}')
        meta=json.loads(folder[0].read_text());cfg=meta['args'];cfg['word_embedding_path']=str(Path(os.environ['BENCH_DATA_DIR'])/'pretrained/wte_pca_500.pt')
        model=Model(SimpleNamespace(**cfg),torch.device('cpu')).float()
        model.load_state_dict(torch.load(folder[0].parent/'checkpoint.pth',map_location='cpu'));model.eval()
        mean=np.array(meta['mean']);std=np.array(meta['std']);mask=inp['horizon']==h
        x=inp['x'][mask];pred=[]
        print(f'Loaded horizon {h} in {time.monotonic()-started:.2f}s; predicting {len(x)} windows', flush=True)
        with torch.no_grad():
            for i in range(0,len(x),256):
                batch=torch.from_numpy(((x[i:i+256]-mean)/std).astype(np.float32))
                p=model(batch)['outputs_time'].cpu().numpy()
                pred.append(p.astype(np.float64)*std+mean)
        out[f'id_{h}']=inp['id'][mask];out[f'pred_{h}']=np.concatenate(pred)
        del model
        print(f'Finished horizon {h} in {time.monotonic()-started:.2f}s', flush=True)
    with open(a.output,'wb') as f:np.savez(f,**out)
if __name__=='__main__':main()
