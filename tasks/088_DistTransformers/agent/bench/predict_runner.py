"""Frozen per-problem prediction interface with diagnostic forward timings."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reference_api import build_model
from validate import COLS


def load_api(code):
    sys.path[:0] = [str(code/'src'), str(code)]
    spec = importlib.util.spec_from_file_location('candidate_api', code/'model_api.py')
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    return api


def snapshot(model):
    if hasattr(model, 'state_dict'):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
    return pickle.dumps(model, protocol=5)


def unchanged(before, model):
    if isinstance(before, bytes):
        return before == pickle.dumps(model, protocol=5)
    after = model.state_dict()
    return set(after) == set(before) and all(torch.equal(v, after[k]) for k, v in before.items())


def timed(fn, phi, z):
    with torch.no_grad():
        start = time.perf_counter_ns()
        out = fn(phi, z)
        elapsed = max((time.perf_counter_ns() - start)*1e-9, 1e-9)
        out = torch.as_tensor(out, dtype=torch.float32).detach()
    return out, elapsed, [elapsed]


def run(model_dir, input_path, output, code):
    cpus = int(os.environ.get('BENCH_CPUS','8'))
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(1)
    seed = int(os.environ.get('BENCH_SEED','1'))
    torch.manual_seed(seed)
    np.random.seed(seed)
    with np.load(input_path, allow_pickle=False) as f:
        ids, phi, z = f['id'], torch.from_numpy(f['phi'].copy()), torch.from_numpy(f['z'].copy())
    # Random, unfitted weights suffice for architecture latency; no evaluation
    # checkpoint is distributed. Forward operator sequence is weight-independent.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        baseline = build_model().train()
    api = load_api(code)
    if getattr(api, 'AMORTISED', 1) != 1:
        raise ValueError('the research task requires amortised inference')
    candidate = api.load_model(model_dir)
    state_before = snapshot(candidate)
    outputs, timings = [], []
    for start in range(0,len(ids),1000):
        ph, ob = phi[start:start+1000], z[start:start+1000]
        out, ct, cr = timed(lambda p,o: api.forward(candidate,p,o), ph, ob)
        _, bt, br = timed(lambda p,o: baseline(p,obs_1=o)[1], ph, ob)
        if out.shape != (len(ph),2,3):
            raise ValueError('expected two log-space Gaussian components, shape (n,2,3)')
        outputs.append(out.cpu().numpy())
        timings.extend([(ct*1000/len(ph),bt*1000/len(ph))]*len(ph))
        print(json.dumps(dict(batch_rows=len(ph),candidate_s=ct,baseline_s=bt,
                              ratio=ct/bt,raw_candidate=cr,raw_baseline=br)),flush=True)
    if not unchanged(state_before, candidate):
        raise ValueError('inference mutated fitted parameters or buffers')
    out = np.concatenate(outputs)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(COLS)
        for i,iid in enumerate(ids):
            writer.writerow([str(iid),*out[i].reshape(-1).tolist(),2,int(getattr(api,'AMORTISED',1)),*timings[i]])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--code',type=Path,default=Path(os.environ.get('BENCH_CODE_DIR','.')))
    a=p.parse_args();run(a.model,a.input,a.output,a.code)
