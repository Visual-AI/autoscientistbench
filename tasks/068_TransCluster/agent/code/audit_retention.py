"""Non-random retention hooks for complete reconstruction of synthetic batches."""
import os,json,hashlib,random
from pathlib import Path
import numpy as np
import torch

class TrainingArchive:
    def __init__(self, out, nsteps, val_tasks, args):
        self.root=Path(out)/'data_archive';self.root.mkdir()
        self.torch_states=np.lib.format.open_memmap(self.root/'torch_rng.npy',mode='w+',dtype=np.uint8,shape=(nsteps,len(torch.get_rng_state())))
        self.python_states=np.lib.format.open_memmap(self.root/'python_rng.npy',mode='w+',dtype=np.uint64,shape=(nsteps,len(random.getstate()[1])))
        self.hashes=(self.root/'batch_sha256.jsonl').open('w')
        torch.save(val_tasks,self.root/'validation_tasks.pt')
        (self.root/'generator.json').write_text(json.dumps({'args':vars(args),'python_rng_version':random.getstate()[0],'numpy_initial_state':repr(np.random.get_state()),'training_batches':nsteps,'tasks_per_batch':args.bsz,'format':'Lossless generative representation: exact pre-generation Torch and Python RNG states for every batch; generator code and dependencies retained. Batch content hashes verify replay.'},indent=2))
    def before(self,step):
        self.torch_states[step]=torch.get_rng_state().numpy()
        self.python_states[step]=random.getstate()[1]
    def after(self,step,x,c):
        h=hashlib.sha256(x.numpy().tobytes()+c.numpy().tobytes()).hexdigest()
        self.hashes.write(json.dumps({'step':step,'sha256':h})+'\n')
        if step in [0,9999]:torch.save((x,c),self.root/f'train_batch_{step}.pt')
        if step%50==0:
            self.hashes.flush();self.torch_states.flush();self.python_states.flush()
            p=self.root/'progress.pending';p.write_text(json.dumps({'last_generated_step':step,'batch_sha256':h}));p.replace(self.root/'progress.json')
    def close(self):
        self.hashes.close();self.torch_states.flush();self.python_states.flush()


def save_eval_input(task_dict, outdir, family):
    out=Path(outdir)/'data_archive';out.mkdir(exist_ok=True)
    torch.save({'tasks':task_dict,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()},out/f'{family}_tasks.pt')


def save_eval_objectives(lloyds,models,outdir,family,label):
    out=Path(outdir)/'data_archive';out.mkdir(exist_ok=True)
    for seed,model in models.items():
        np.savez(out/f'{family}_{seed}_objectives.npz',model=model.detach().cpu().numpy(),lloyds=lloyds[seed].detach().cpu().numpy(),label=label)
