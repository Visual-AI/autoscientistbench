"""Frozen-forecast data entry and causal train/predict protocol for ResCP."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
from reservoir_conformal_prediction.src.torch_reservoir_computing.reservoir import Reservoir
from reservoir_conformal_prediction.src.utils.utils import minimize_intervals
import sampling_core
from sampler_slices import install_sampling

install_sampling(sampling_core)
os.environ['REPRO_SIM_GEMM'] = '0'
THREADS = min(4, max(1, int(os.environ.get('BENCH_CPUS', '1'))))
EPS = 5e-8

def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)

def normalized_states(reservoir, x):
    x = torch.where(x.isnan(), torch.tensor(0.), x)
    state = reservoir.get_states(x.permute(1, 0, 2), bidir=False).permute(1, 0, 2)
    norm = torch.linalg.norm(state, dim=-1, keepdim=True)
    return torch.where(~torch.isclose(norm, torch.zeros_like(norm)), state/norm, state)

def train_cell(job):
    cell, data, out, seed = job
    torch.set_num_threads(THREADS)
    cfg = json.loads((Path(__file__).parent/'method_config.json').read_text())[cell]
    a = np.load(Path(data)/'train'/(cell+'.npz'), allow_pickle=False)
    r = a['residual'].copy()
    if cfg['clip']:
        lower, upper = np.nanquantile(r[a['cal_index']], [.005, .995], axis=0)
        r = np.clip(r, lower, upper).astype(np.float32)
    else:
        lower = np.full(r.shape[1:], -np.inf); upper = -lower
    fit = r[a['fit_index']]
    # TSL StandardScaler graph axes and float32 population reductions.
    bias = np.mean(fit, axis=(0,1), keepdims=True)
    scale = np.std(fit, axis=(0,1), keepdims=True)
    scale[np.isclose(scale, 0, atol=EPS)] = 1.
    reservoir = Reservoir(seed=seed, **cfg['reservoir'])
    z = (tensor(r)-tensor(bias))/(tensor(scale)+EPS)
    states = torch.cat([torch.zeros((1,r.shape[1],cfg["reservoir"]["n_internal_units"])), normalized_states(reservoir,z[1:])])
    idx = a['cal_index']
    obj = dict(cell=cell, seed=seed, cfg=cfg, bias=bias, scale=scale, lower=lower, upper=upper,
               internal=reservoir._internal_weights, input_weights=reservoir._input_weights,
               cal_states=states[idx-cfg['delay']-1], cal_residual=tensor(r[idx]))
    torch.save(obj, Path(out)/(cell+'.pt'))
    print('calibrated', cell, flush=True)

def predict_cell(job):
    cell, model, input_path, seed = job
    torch.set_num_threads(THREADS)
    obj = torch.load(Path(model)/(cell+'.pt'), map_location='cpu', weights_only=False)
    cfg = obj['cfg']
    a = np.load(input_path, allow_pickle=False)
    raw = a[cell+'__residual']
    r = np.clip(raw, obj['lower'], obj['upper']).astype(np.float32)
    # Recreate the released reservoir using saved weights. No fitting on the query block.
    reservoir = object.__new__(Reservoir)
    reservoir.__dict__.update(_n_internal_units=cfg['reservoir']['n_internal_units'],
        _noise_level=cfg['reservoir']['noise_level'], _leak=cfg['reservoir']['leak'],
        _internal_weights=obj['internal'], _input_weights=obj['input_weights'])
    z = (tensor(r)-tensor(obj["bias"]))/(tensor(obj["scale"])+EPS)
    states = torch.cat([torch.zeros((1,r.shape[1],cfg["reservoir"]["n_internal_units"])), normalized_states(reservoir,z[1:])])
    idx = a[cell+'__query']
    x = states[idx-cfg['delay']-1]
    point = tensor(a[cell+'__point'])
    # A current target is only needed AFTER its interval. The final target is unavailable.
    # eta=0 in every released setting, and only prior residuals enter future pools.
    observed = torch.zeros_like(point)
    for i, t in enumerate(idx):
        if t < len(r): observed[i] = tensor(r[t])
    truth_for_update = point + observed
    cls = sampling_core.ConformalResidualSampler if point.shape[1] == 1 else sampling_core.ConformalResidualSamplerParallel
    sampler = cls(cal_residuals=obj['cal_residual'].clone(), reservoir=None,
                  alpha=.1, **cfg['sampler'])
    sampler.cal_states = obj['cal_states'].clone()
    sampler.in_sweep = True
    # Separate sampling RNG from the length-dependent reservoir-noise draw stream.
    # This keeps the random algorithm and makes prefix predictions reproducible.
    torch.manual_seed(seed)
    with torch.no_grad():
        quantiles = sampler(x, point, truth_for_update)
        low, high = minimize_intervals(quantiles, point, point)
    ids = a[cell+'__id'].reshape(-1)
    return list(zip(ids.tolist(), low.numpy().reshape(-1).tolist(), high.numpy().reshape(-1).tolist()))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict'])
    p.add_argument('--out');p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    seed=int(os.environ.get('BENCH_SEED','1'))
    cells=sorted(json.loads((Path(__file__).parent/'method_config.json').read_text()))
    workers=min(len(cells),max(1,int(os.environ.get('BENCH_CPUS','1'))//THREADS))
    if a.stage=='train':
        Path(a.out).mkdir(parents=True,exist_ok=True)
        jobs=[(c,os.environ['BENCH_DATA_DIR'],a.out,seed) for c in cells]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(train_cell,jobs))
    else:
        jobs=[(c,a.model,a.input,seed) for c in cells]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(predict_cell,jobs))
        with open(a.output,'w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','lower','upper'])
            for cell_rows in rows:w.writerows(cell_rows)
if __name__=='__main__':main()
