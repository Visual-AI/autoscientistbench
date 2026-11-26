"""Protocol entry points for the unchanged Bunny DDPM."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from trainers import DDPMTrainer
from utils.constraints import MeshConstraintProjector

def main(a):
    seed = int(os.environ.get('BENCH_SEED', '1'))
    torch.set_num_threads(min(8, int(os.environ.get('BENCH_CPUS', '16'))))
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = Path(os.environ['BENCH_DATA_DIR']) / 'train'
    points = torch.from_numpy(np.load(data / 'samples.npy')).clone()
    projector = MeshConstraintProjector(str(data / 'mesh.obj'), 'cpu')
    trainer = DDPMTrainer(points.squeeze(), project_x0_sample=True,
        timesteps=250, batch_size=64, lr=1e-3, hidden_dim=128,
        time_embed_dim=32, time_concat=True, mesh=True, projector=projector,
        constraints_dict={'bunny':str(data / 'mesh.obj')})
    if a.cmd == 'train':
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        trainer.train(epochs=200)
        trainer.save_checkpoint(str(out / 'model.pth'), epoch=200)
        (out / 'config.json').write_text(json.dumps({'seed':seed,'epochs':200,'timesteps':250,'hidden_dim':128,'batch_size':64,'lr':.001,'sigma':.0005}))
    else:
        with np.load(a.input, allow_pickle=False) as z:
            ids = z['id']
        trainer.load_checkpoint(str(Path(a.model) / 'model.pth'), map_location='cpu', load_optimizer=False)
        trainer.denoiser.eval()
        with torch.no_grad():
            raw, _ = trainer.sample(num_samples=len(ids))
        # Preserve the authors' nonfinite outputs. The trusted scorer reprojects.
        with open(a.output, 'wb') as f:
            np.savez(f, id=ids, samples=np.asarray(raw))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest='cmd', required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    q=sp.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    main(p.parse_args())
