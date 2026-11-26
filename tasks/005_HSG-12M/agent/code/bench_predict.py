import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from bench_train import make_model
from packed import PackedCollator

ap = argparse.ArgumentParser()
for name in ('model','input','output'): ap.add_argument('--'+name, type=Path, required=True)
a = ap.parse_args()
torch.set_num_threads(min(20, int(os.environ.get('BENCH_CPUS','128')))); torch.set_num_interop_threads(1)
seed = int(os.environ.get('BENCH_SEED','1')); torch.manual_seed({1:42,2:2025,3:666}.get(seed,seed))
model = make_model(); model.load_state_dict(torch.load(a.model/'weights.pt', map_location='cpu', weights_only=True)); model.eval()
card = json.loads((a.model/'model_card.json').read_text())
collate = PackedCollator(a.input); n = len(collate.ids)
pred = np.full(n, -1, dtype=np.int64)
with torch.inference_mode():
    for rank in (0, 1):
        positions = list(DistributedSampler(range(n), num_replicas=2, rank=rank, shuffle=False, drop_last=False))
        for start in range(0, len(positions), 6000):
            pos = np.asarray(positions[start:start+6000])
            pred[pos] = model(collate(pos)).argmax(dim=1).cpu().numpy()
assert (pred >= 0).all()
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['id','pred','parameters','steps','model_sha256','epochs'])
    for i, y in zip(collate.ids, pred):
        w.writerow([i, int(y), card['trainable_parameters'], card['optimizer_steps'], card['weights_sha256'], card['epochs_started']])
