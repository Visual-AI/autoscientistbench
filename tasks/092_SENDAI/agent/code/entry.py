"""Protocol entry: fixed sensor observations, three independent runs, label-free inference."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import config as c
from SENDAI import SHRED,DASHRED,FullDASHRED,SHREDDataset,fix_bad_frames,create_time_delay_dataset
from SENDAI import train_shred,train_dashred,train_hierarchical_hf
from fast import install,install_rng,install_previous

class SparseDataset(SHREDDataset):
    def __init__(self,sensors,locs,scaler_sensor,scaler_state):
        # SHREDDataset needs a state-shaped placeholder; only sensor columns are consumed
        # in the real-period training loop. No real full field is loaded or reconstructed.
        idx=locs[:,0]*64+locs[:,1]
        states=np.zeros((len(sensors),4096),dtype=np.float64)
        states[:,idx]=sensors[:,-1,:]
        super().__init__(sensors,states,idx,scaler_sensor,scaler_state)


def fit(a):
    seed=int(a.seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,int(os.environ['SENDAI_THREADS'])))
    torch.set_num_interop_threads(1)
    install();install_rng();install_previous()
    data=Path(os.environ['BENCH_DATA_DIR'])/'train'
    sim=np.load(data/'simulation.npz')['fields'];obs=np.load(data/'observations.npz')
    sim,_=fix_bad_frames(sim);locs=obs['sensor_locs'];idx=locs[:,0]*64+locs[:,1]
    sensors,states=create_time_delay_dataset(sim,locs,c.LAGS);n=int(len(sensors)*.8)
    tr=SHREDDataset(sensors[:n],states[:n],idx,fit_scaler=True)
    va=SHREDDataset(sensors[n:],states[n:],idx,tr.scaler_sensor,tr.scaler_state)
    real=SparseDataset(obs['sensors'],locs,tr.scaler_sensor,tr.scaler_state)
    tl=DataLoader(tr,batch_size=c.BATCH_SIZE,shuffle=True)
    vl=DataLoader(va,batch_size=c.BATCH_SIZE)
    rl=DataLoader(real,batch_size=c.BATCH_SIZE,shuffle=True)
    shred=SHRED(c.N_SENSORS,c.LAGS,c.HIDDEN_SIZE,4096,num_layers=c.NUM_LSTM_LAYERS,decoder_layers=c.DECODER_LAYERS)
    shred=train_shred(shred,tl,vl,epochs=c.SHRED_EPOCHS,lr=c.LR,patience=c.SHRED_PATIENCE)
    da=DASHRED(shred,freeze_decoder=False)
    da=train_dashred(da,tl,rl,idx,epochs=c.DASHRED_EPOCHS,lr=c.LR,patience=c.DASHRED_PATIENCE,gan_epochs=c.GAN_EPOCHS)
    model=FullDASHRED(da,c.N_SENSORS,4096,idx,64,64,n_hf_layers=c.N_PEEL_LAYERS,use_inr=c.USE_INR,inr_config=c.INR_CONFIG)
    torch.set_num_threads(int(os.environ['SENDAI_THREADS']))
    model,hist,freq=train_hierarchical_hf(model,rl,64,64,peel_config=c.PEEL_CONFIG,max_freq=c.MAX_TARGET_FREQ,exclusion_radius=2.0,lambda_smooth=c.INR_CONFIG['lambda_smooth'],smooth_type=c.INR_CONFIG['smooth_type'],warmup=c.HF_WARMUP,base_lr=c.LR)
    torch.save({'full_model':model.state_dict(),'sensor_locs':locs,'scaler_sensor':tr.scaler_sensor,'scaler_state':tr.scaler_state,'seed':seed},a.out)
    print('Saved',a.out,flush=True)


def train(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    build=out/'build'
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_extension.py')),str(build)],check=True)
    outer=int(os.environ.get('BENCH_SEED','1'))
    # Separate RNG streams for the three independent repetitions, not an ensemble.
    seeds=[3*outer+i for i in range(3)]
    threads=max(1,min(32,int(os.environ.get('BENCH_CPUS','4'))//3))
    env=dict(os.environ,SENDAI_BUILD_DIR=str(build),SENDAI_THREADS=str(threads))
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]=str(threads)
    children=[]
    try:
        for i,seed in enumerate(seeds):
            log=(out/f'fit_{i}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'fit','--seed',str(seed),'--out',str(out/f'model_{i}.pt')],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((p,log))
        codes=[p.wait() for p,_ in children]
        if any(codes):raise RuntimeError(f'independent training failed: {codes}; see fit logs')
    finally:
        for p,log in children:
            if p.poll() is None:p.terminate()
        for p,log in children:p.wait();log.close()
    card={'format':'sendai-sparse/v1','sensor_locs':np.load(Path(os.environ['BENCH_DATA_DIR'])/'train/observations.npz')['sensor_locs'].tolist(),'lags':c.LAGS,'seeds':seeds,'outer_seed':outer,'replicates':3}
    (out/'model_card.json').write_text(json.dumps(card,indent=2)+'\n')


def predict(a):
    torch.set_num_threads(min(32,int(os.environ.get('BENCH_CPUS','4'))));torch.set_num_interop_threads(1)
    install()
    inp=np.load(a.input);locs=inp['sensor_locs'];idx=locs[:,0]*64+locs[:,1]
    predictions=[]
    for i in range(3):
        checkpoint=torch.load(Path(a.model)/f'model_{i}.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(locs,checkpoint['sensor_locs'])
        shred=SHRED(c.N_SENSORS,c.LAGS,c.HIDDEN_SIZE,4096,num_layers=c.NUM_LSTM_LAYERS,decoder_layers=c.DECODER_LAYERS)
        model=FullDASHRED(DASHRED(shred),c.N_SENSORS,4096,idx,64,64,n_hf_layers=c.N_PEEL_LAYERS,use_inr=c.USE_INR,inr_config=c.INR_CONFIG)
        model.load_state_dict(checkpoint['full_model'],strict=True);model.eval()
        ds=SparseDataset(inp['sensors'],locs,checkpoint['scaler_sensor'],checkpoint['scaler_state'])
        output=[]
        with torch.no_grad():
            for sensors,_,current in DataLoader(ds,batch_size=64):
                total,*_=model(sensors,current,use_hf=True);output.append(total.numpy())
        predictions.append(checkpoint['scaler_state'].inverse_transform(np.vstack(output)).reshape(-1,64,64))
    with open(a.output,'wb') as f:np.savez(f,ids=inp['ids'],pred=np.stack(predictions),sensor_locs=locs,lags=np.array(c.LAGS))

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train');p.add_argument('--out',required=True)
    p=sub.add_parser('fit');p.add_argument('--out',required=True);p.add_argument('--seed',required=True)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args();globals()[a.command](a)
