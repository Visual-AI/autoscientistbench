"""Offline train/predict protocol for the unchanged subject encoder."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/name)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def train_subject(subject, data, out):
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    from fmri_mapping.embedding import ops
    from fmri_mapping.embedding.mlp import ResidualMLP
    from fmri_mapping.utils.logger import WandbLogger
    threads = int(os.environ['WORKER_CPUS'])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    threadpool_limits(threads)
    seed = int(os.environ['BENCH_SEED'])
    torch.manual_seed(seed); np.random.seed(seed)
    m = module('1_within_subject_linear.py')
    df = pd.read_csv(data/f'stimulus_{subject}.csv')
    ops.get_resource = lambda *args, **kwargs: df.copy()
    observed = df[df.exists].sort_values('subject_index')
    with np.load(data/f'subject_{subject}.npz', allow_pickle=False) as p:
        raw = p['betas']; run = p['run']
        assert np.array_equal(observed.id.to_numpy(), p['id'])
    print(f'Subject {subject}: raw loaded, {raw.shape}', flush=True)
    qcuts = ops.int16_scaled_quantiles(raw, [.0005, .9995], 300.)
    betas = raw.astype(np.float32); del raw
    betas /= 300.
    betas = np.clip(betas, qcuts[0], qcuts[1], dtype=np.float32)
    center = np.mean(betas, axis=0)
    betas -= center
    categories = np.unique(run)
    C = pd.get_dummies(run, drop_first=False).to_numpy(dtype=np.float32)
    m.get_confound_matrix = lambda subject: C
    repetitions = ops.split_repetitions(subject, shuffle_indexes=True)
    print(f'Subject {subject}: linear fit', flush=True)
    Z, matrices = m.compute_linear_within_subject_pipeline(subject, betas, observed, repetitions,
        n_components_pca=768, n_components_mcca=128, progress=False)
    print(f'Subject {subject}: linear complete', flush=True)
    matrices.pop('C')
    matrices.update(qcuts=qcuts, center=center, run_categories=categories)
    del betas, C
    # Same two-model initialization, shuffle, and seed reset as the reproduction.
    mlp_threads = min(4, threads)
    torch.set_num_threads(mlp_threads); threadpool_limits(mlp_threads)
    n = module('2_within_subject_mlp.py')
    torch.manual_seed(seed); np.random.seed(seed)
    repetitions = ops.split_repetitions(subject, shuffle_indexes=True, how='outer')
    views = ops.split_betas_by_subject_index(Z.astype(np.float32), repetitions, inpute='mean')
    views = ops.to_tensor(*views, device='cpu', dtype=torch.float)
    parameters = dict(d_in=128, d_hidden=768, depth=1, dropout=.1, initial_alpha=.25, alpha_trainable=True)
    model = ResidualMLP(dtype=torch.float, **parameters)
    decoder = ResidualMLP(dtype=torch.float, **parameters)
    training = dict(temperature=.09, lr=1e-4, weight_decay=1e-4, optimizer='adam', seed=seed,
        batch_size=512, steps=2000, eval_every=200, lambda_pull=.5, lambda_geom=0.,
        lambda_identity_preserve=0., lambda_circle=0., lambda_vsp=0., lambda_nce=1., lambda_loo=0., lambda_mse=0.)
    logger = WandbLogger(mode='disabled', log_frequency=10)
    model, _ = n.train_mlp_triplets(*views, model=model, decoder_model=decoder, device='cpu', logger=logger, **training)
    logger.finish()
    torch.save(dict(matrices=matrices, model_parameters=parameters, model_state_dict=model.state_dict(),
        training_parameters=training, subject=subject), out/f'subject_{subject}.pt')
    print(f'Finished subject {subject}', flush=True)

def train(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ['BENCH_DATA_DIR'])/'train'
    subjects = sorted(int(p.stem.split('_')[-1]) for p in data.glob('subject_*.npz'))
    cpus = int(os.environ.get('BENCH_CPUS', '1'))
    jobs = min(len(subjects), cpus)
    threads = min(16, max(1, cpus//jobs))
    def run(subject):
        env = dict(os.environ, WORKER_CPUS=str(threads))
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            env[key] = str(threads)
        with (out/f'subject_{subject}.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(ROOT/'main.py'), 'worker', '--subject', str(subject), '--out', str(out)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run, subjects))
    (out/'model_card.json').write_text(json.dumps(dict(format='subject-encoder/v1', subjects=subjects,
        embedding_dim=128, seed=int(os.environ['BENCH_SEED'])), indent=2)+'\n')

def predict_worker(args):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from fmri_mapping.embedding.mlp import ResidualMLP
    threads = min(16, int(os.environ.get('BENCH_CPUS', '1')))
    torch.set_num_threads(threads); torch.set_num_interop_threads(1); threadpool_limits(threads)
    torch.manual_seed(int(os.environ.get('BENCH_SEED','1')))
    result = {}
    for path in sorted(Path(args.input).glob('subject_*.npz')):
        subject = int(path.stem.split('_')[-1])
        if subject != args.subject:
            continue
        saved = torch.load(Path(args.model)/f'subject_{subject}.pt', map_location='cpu', weights_only=False)
        m = saved['matrices']
        model = ResidualMLP(**saved['model_parameters']); model.load_state_dict(saved['model_state_dict']); model.eval()
        with np.load(path, allow_pickle=False) as inputs:
            ids, raw, runs = inputs['id'], inputs['betas'], inputs['run']
        outputs = []
        with torch.no_grad():
            for start in range(0, len(ids), 256):
                end = min(start+256,len(ids))
                X = raw[start:end].astype(np.float32); X /= 300.
                X = np.clip(X, m['qcuts'][0], m['qcuts'][1], dtype=np.float32)
                X -= m['center']
                C = (runs[start:end,None] == m['run_categories'][None,:]).astype(np.float32)
                X -= C @ m['B']
                X *= m['W_rel'][None,:]
                X = (X-m['mu_pca'][None,:]) @ m['W_pca']
                X = X @ m['W_mcca'] + m['b_mcca'][None,:]
                outputs.append(model(torch.from_numpy(X.astype(np.float32))).numpy())
        result[f'id_{subject}'] = ids
        result[f'embedding_{subject}'] = np.concatenate(outputs)
    with open(args.output,'wb') as stream:
        np.savez(stream, **result)

def predict(args):
    import tempfile
    import numpy as np
    paths = sorted(Path(args.input).glob('subject_*.npz'))
    subjects = [int(p.stem.split('_')[-1]) for p in paths]
    cpus = int(os.environ.get('BENCH_CPUS','1'))
    jobs = min(len(subjects), cpus)
    threads = min(16, max(1, cpus//jobs))
    with tempfile.TemporaryDirectory(prefix='fmri-predict-') as tmp:
        def run(subject):
            env = dict(os.environ, BENCH_CPUS=str(threads))
            for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
                env[key] = str(threads)
            subprocess.run([sys.executable, str(ROOT/'main.py'), 'predict-worker', '--model', args.model,
                '--input', args.input, '--output', str(Path(tmp)/f'{subject}.npz'), '--subject', str(subject)], env=env, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(run, subjects))
        result = {}
        for subject in subjects:
            with np.load(Path(tmp)/f'{subject}.npz',allow_pickle=False) as part:
                result.update({key:part[key] for key in part.files})
        with open(args.output,'wb') as stream:
            np.savez(stream, **result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train','predict','worker','predict-worker'])
    parser.add_argument('--out'); parser.add_argument('--model'); parser.add_argument('--input'); parser.add_argument('--output'); parser.add_argument('--subject',type=int)
    args = parser.parse_args()
    if args.stage == 'train': train(args)
    elif args.stage == 'predict': predict(args)
    elif args.stage == 'predict-worker': predict_worker(args)
    else: train_subject(args.subject, Path(os.environ['BENCH_DATA_DIR'])/'train', Path(args.out))
