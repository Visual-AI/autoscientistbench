"""Protocol adapter for independent GPLFR fits and raw point-grid prediction."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

SUBSETS = ['multi-partial', 'multi-complete', 'single-complete']
CODE = Path(__file__).resolve().parent

def worker(a):
    import numpy as np
    import torch
    import thousandworlds as tw
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.set_num_interop_threads(1)
    if a.cmd == 'fit':
        from thousandworlds.models.gplfr import GPLFR
        import pyro
        data = Path(a.data) / 'train' / a.subset
        with (data / 'stats.pkl').open('rb') as f:
            stats = pickle.load(f)
        with np.load(data / 'train.npz', allow_pickle=False) as raw:
            X = tw.transform_inputs(raw['X'], stats)
            y = np.transpose(tw.normalise_spectral(raw['Y_spectral'], raw['raw_field_names'].tolist(), stats), (0, 2, 1))
            hp = json.loads((CODE / 'configs' / f'{a.subset}.json').read_text())
            hp.pop('n_samples')
            hp.update(hp.pop('optimizer'))
            model = GPLFR(**hp, dtype=torch.float64, device='cpu')
            model.fit(torch.from_numpy(X), torch.from_numpy(raw['s'].copy()), torch.from_numpy(y),
                      field_mask=torch.from_numpy(raw['field_mask']), sh_mask=torch.from_numpy(raw['sh_mask']),
                      field_names=raw['raw_field_names'].tolist(), seed=a.seed,
                      n_sim_types=int(raw['n_sim_types']))
            torch.save({'model': model, 'pyro_params': pyro.get_param_store().get_state(),
                        'stats': stats, 'raw_field_names': raw['raw_field_names'].tolist(),
                        'sh_mask': raw['sh_mask'], 'seed': a.seed}, a.out)
    else:
        from thousandworlds.run_model import _gplfr_decode_public_predictions
        state = torch.load(a.model, weights_only=False, map_location='cpu')
        with np.load(a.input, allow_pickle=False) as inp:
            X = inp[a.subset + '__X']; s = inp[a.subset + '__s']
        data = SimpleNamespace(stats=state['stats'], sh_mask=state['sh_mask'], inverse_sht=tw.load_inverse_sht_matrix(),
                               spectral_bundle=SimpleNamespace(raw_field_names=state['raw_field_names']),
                               grid_bundle=SimpleNamespace(X_test=X))
        coeff = state['model'].predict(torch.from_numpy(tw.transform_inputs(X, state['stats'])),
                                       torch.from_numpy(s.copy())).detach().cpu().numpy()[None]
        pred = _gplfr_decode_public_predictions(coeff, data)[0]
        with open(a.output, 'wb') as f:
            np.save(f, pred)

def run(a):
    import numpy as np
    cpus = max(1, int(os.environ.get('BENCH_CPUS', '64')))
    threads = min(4, cpus)
    workers = min(10, max(1, cpus // threads))
    env = dict(os.environ)
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[k] = str(threads)
    # Five independent fits per evaluation, not a prediction ensemble. Seed 1 uses 0..4.
    group = int(os.environ.get('BENCH_SEED', str(getattr(a, 'seed', 1))))
    if a.cmd == 'train':
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        seeds = [(5 * (group - 1) + i) % (2**32) for i in range(5)]
        tasks = [(s, i, seed) for s in SUBSETS for i, seed in enumerate(seeds)]
        def execute(t):
            s,i,seed=t
            with (out / f'{s}_{i}.log').open('w') as log:
                subprocess.run([sys.executable, '-u', str(CODE/'main.py'), 'fit', '--data', a.data,
                                '--subset',s,'--seed',str(seed),'--out',str(out/f'{s}_{i}.pt')],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('fitted',s,i,flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(execute,tasks))
        (out/'metadata.json').write_text(json.dumps({'algorithm_seeds':seeds,'bench_seed':group,'threads_per_worker':threads}))
    else:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='tw_predict_') as tmp:
            tmp=Path(tmp)
            def execute(t):
                s,i=t
                subprocess.run([sys.executable,str(CODE/'main.py'),'point','--model',str(Path(a.model)/f'{s}_{i}.pt'),
                                '--subset',s,'--input',a.input,'--output',str(tmp/f'{s}_{i}.npy')],env=env,check=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(execute,[(s,i) for s in SUBSETS for i in range(5)]))
            payload={}
            with np.load(a.input,allow_pickle=False) as inp:
                for s in SUBSETS:
                    payload[s+'__id']=inp[s+'__id']
                    payload[s+'__pred']=np.stack([np.load(tmp/f'{s}_{i}.npy') for i in range(5)])
            with open(a.output,'wb') as f:np.savez(f,**payload)

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['train','predict','fit','point'])
    for arg in ['data','out','model','input','output','subset']:ap.add_argument('--'+arg)
    ap.add_argument('--seed',type=int,default=1);a=ap.parse_args()
    worker(a) if a.cmd in ['fit','point'] else run(a)
