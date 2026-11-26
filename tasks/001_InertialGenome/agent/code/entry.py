"""Benchmark data entry, seeded training, and coordinate-only inference."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import InertialTransformer_Genome as m
import cpu_opt

MAX_LEN=2400

def model_at(out):
    return m.InertialTransformer_Modeling(num_atom=MAX_LEN,latent_dim=48,num_heads=4,
        num_layers=1,max_seq_len=MAX_LEN,max_coordinate=4,num_sigma=12,
        num_gaussian_kernels=8,use_edge_type=True,rope_theta=5,recycle=1,scale=4,
        anchor_dir=str(out/'anchors'),apply_index_pos_emb=True,use_qk_layernorm=True,
        apply_selective_rope='full',use_rope=True,use_nystrom=True)

def dataset(inp,labels=None):
    return m.ChromosomeDataset(str(inp),str(labels) if labels else None,[],
        max_len=MAX_LEN,augment=False,use_inertial_frame=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict'])
    p.add_argument('--out',type=Path);p.add_argument('--model',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    seed=int(os.environ.get('BENCH_SEED','1'));threads=int(os.environ.get('BENCH_CPUS','8'))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    cpu_opt.install(m);cpu_opt.enable_rank_cache(m)
    if a.stage=='train':
        out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
        assert not (out/'best_model.pth').exists(), 'use a fresh output directory'
        model=model_at(out)
        data=Path(os.environ['BENCH_DATA_DIR'])
        train=DataLoader(dataset(data/'train/input.npz',data/'train/labels.npz'),batch_size=4,
            shuffle=True,num_workers=0,collate_fn=cpu_opt.rank_collate)
        val=DataLoader(dataset(data/'val/input.npz',data/'val/labels.npz'),batch_size=4,
            shuffle=False,num_workers=0,collate_fn=cpu_opt.rank_collate)
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-2)
        best,history=m.run_training(model,train,val,optimizer,torch.device('cpu'),epochs=50,
            save_model=str(out/'best_model.pth'),save_picture=str(out/'training.png'))
        (out/'history.json').write_text(json.dumps({'seed':seed,'best_val_dscc':best,
            'best_epoch':int(np.argmax(history['val_dscc']))+1,'history':history},indent=2))
    else:
        model=model_at(a.model.resolve())
        model.load_state_dict(torch.load(a.model/'best_model.pth',map_location='cpu',weights_only=True))
        model.eval();ds=dataset(a.input)
        loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('w',newline='') as f,torch.no_grad():
            writer=csv.writer(f);writer.writerow(['id','slot','x','y','z'])
            for i,batch in enumerate(loader):
                coords,_=model(batch['idx'],batch['coords'])
                for slot,xyz in enumerate(coords[0].numpy()):
                    writer.writerow([ds.ids[i],slot,*[float(x) for x in xyz]])
if __name__=='__main__':main()
