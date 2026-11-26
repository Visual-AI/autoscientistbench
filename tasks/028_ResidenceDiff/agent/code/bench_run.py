"""Benchmark data/save entry points for the unchanged RTA reconstruction."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import csv
import numpy as np
from pdda.profiles import smooth_profile
from rta_common import recompute_from_arrays, wall_allowed_mask, rebuild_analysis


def train(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for directory in sorted((Path(args.data)/'train').iterdir()):
        if not directory.is_dir(): continue
        t=time.monotonic(); meta=json.loads((directory/'meta.json').read_text())
        with np.load(directory/'statistics.npz',allow_pickle=False) as z: e=dict(z)
        e['diffusivity_sems']=e.pop('diffusivity_sem')
        e['first_step_exit_fractions']=e.pop('first_step_exit_fraction')
        e['terminal_censored_fractions']=e.pop('terminal_censored_fraction')
        arrays=[np.load(directory/name,mmap_mode='r') for name in ['cell_keys.npy','observed_lengths.npy','observed_counts.npy']]
        print(f'Recomputing {directory.name} residence times and excess NLL',flush=True)
        d,nll,diag=recompute_from_arrays(e['interval_widths'],e['positions'],meta['sampling_period_ps'],arrays[0],meta['n_centers'],arrays[1],arrays[2],allowed=wall_allowed_mask(e['interval_widths'],e['positions'],meta['domains']))
        analysis=rebuild_analysis(e,d,nll,meta['sampling_period_ps'])
        minima=analysis.constant_width_minima()
        selection=analysis.select_profile(plateau_tolerance=0.05)
        profile=analysis.profile_at_width(7.5)
        # Same fit block as notebooks/analysis_helpers.prepare_propagator_inputs.
        values=np.column_stack((profile.positions,profile.diffusivity))
        values=values[np.abs(values[:,0])<=0.5*meta['box_length']]
        fitted=smooth_profile(values[:,0],values[:,1],positive=True,symmetry_center=0.0,smoothing_length=1.0,plateau=True,plateau_tolerance=0.05,extrapolate=True)
        with np.load(directory/'context.npz',allow_pickle=False) as z: grid=z['positions']
        np.savez(out/f'{directory.name}.npz',positions=grid,diffusivity=fitted(grid),raw_diffusivity=d,excess_nll=nll)
        diag.update(selection=selection.summary(),selected_width=float(selection.interval_width),minima=[float(p.interval_width) for p in minima],elapsed_s=time.monotonic()-t)
        (out/f'{directory.name}.json').write_text(json.dumps(diag,indent=2)+'\n')
        print(json.dumps(diag),flush=True)
    (out/'config.json').write_text(json.dumps({'seed':args.seed,'interval_width':7.5,'smoothing_length':1.0,'plateau_tolerance':0.05})+'\n')


def predict(args):
    with np.load(args.input,allow_pickle=False) as z: inp=dict(z)
    system=str(inp['system'].item())
    if system not in ('popc_water','sc_water'): raise ValueError('unknown physical system')
    with np.load(Path(args.model)/f'{system}.npz',allow_pickle=False) as z:
        if not np.array_equal(z['positions'],inp['positions']): raise ValueError('unexpected spatial grid')
        diffusion=z['diffusivity'].copy()
    with open(args.output,'w',newline='') as f:
        w=csv.writer(f,lineterminator='\n'); w.writerow(['id','diffusivity'])
        w.writerows((str(i),format(float(d),'.17g')) for i,d in zip(inp['id'],diffusion,strict=True))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='stage',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--out',required=True); tr.add_argument('--data',required=True); tr.add_argument('--seed',type=int,default=int(os.environ.get('BENCH_SEED','1')))
    pr=sub.add_parser('predict'); pr.add_argument('--model',required=True); pr.add_argument('--input',required=True); pr.add_argument('--output',required=True)
    a=p.parse_args(); seed=getattr(a,'seed',int(os.environ.get('BENCH_SEED','1'))); random.seed(seed); np.random.seed(seed)
    (train if a.stage=='train' else predict)(a)
if __name__=='__main__': main()
