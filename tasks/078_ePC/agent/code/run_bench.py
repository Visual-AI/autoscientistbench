import hashlib
"""Offline protocol adapter; predictive-coding core is the authors' unchanged code."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def train_one(out, slot, seed):
    import lightning as L
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from get_arch import get_architecture
    from pc_variants import get_pc_variant

    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.set_num_interop_threads(1)
    L.seed_everything(seed, workers=True)

    class Data(L.LightningDataModule):
        batch_size = 64
        def setup(self, stage):
            z = np.load(Path(os.environ['BENCH_DATA_DIR']) / 'train/train.npz', allow_pickle=False)
            # Same float32 operations as ToTensor -> Normalize(0.5, 0.5) -> flatten.
            x = torch.from_numpy(z['x']).float().div(255).sub_(0.5).div_(0.5).flatten(1)
            self.train_set = TensorDataset(x, torch.from_numpy(z['y']).long())
        def train_dataloader(self):
            return DataLoader(self.train_set, batch_size=64, shuffle=True, drop_last=True)
        def val_dataloader(self):
            return []
        def on_after_batch_transfer(self, batch, dataloader_idx):
            return {'img': batch[0], 'y': torch.nn.functional.one_hot(batch[1], num_classes=10).float()}

    class Evidence(L.Callback):
        def on_train_start(self, trainer, pl_module):
            self.started = time.monotonic()
            self.rows = []
        def on_train_epoch_end(self, trainer, pl_module):
            r = {'epoch': trainer.current_epoch + 1, 'global_step': trainer.global_step,
                 'elapsed_s': time.monotonic() - self.started}
            self.rows.append(r)
            (out / f'epochs{slot}.json').write_text(json.dumps(self.rows, indent=2))
            print(json.dumps({'replica': slot, **r}), flush=True)

    data = Data()
    pc = get_pc_variant('EO', True)(get_architecture('FashionMNIST', use_CELoss=True),
                                   iters=4, e_lr=0.003, w_lr=5e-5)
    trainer = L.Trainer(accelerator='cpu', devices=1, max_epochs=14, inference_mode=False,
                        logger=False, enable_checkpointing=False, callbacks=[Evidence()],
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(pc, datamodule=data)
    np.savez(out / f'replica{slot}.npz', **{k: v.detach().cpu().numpy() for k, v in pc.state_dict().items()})
    print(f'Finished replica {slot}, RNG seed {seed}', flush=True)


def train(out):
    out.mkdir(parents=True, exist_ok=True)
    cpus = max(1, int(os.environ.get('BENCH_CPUS', '16')))
    threads = min(2, cpus)
    workers = min(5, max(1, cpus // threads))
    base = int(os.environ.get('BENCH_SEED', '1'))
    def run(slot):
        seed = (5 * (base - 1) + slot) % (2 ** 32)
        env = os.environ.copy()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
            env[name] = str(threads)
        with (out / f'train{slot}.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'train-one', '--out', str(out),
                            '--slot', str(slot), '--seed', str(seed)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return {'slot': slot, 'rng_seed': seed, 'weights': f'replica{slot}.npz', 'sha256': hashlib.sha256((out/f'replica{slot}.npz').read_bytes()).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        cards = list(pool.map(run, range(5)))
    card = {'format': 'epc-mlp-v1', 'dimensions': [784, 128, 128, 128, 10], 'activation': 'gelu',
            'parameters_per_model': 134794, 'replicas': cards,
            'training': {'rule': 'error-based predictive coding', 'epochs': 14, 'batch_size': 64,
                         'iters': 4, 'e_lr': 0.003, 'w_lr': 5e-5, 'bench_seed': base}}
    (out / 'model_card.json').write_text(json.dumps(card, indent=2) + '\n')


def predict(model, input_path, output):
    import numpy as np
    import torch
    from get_arch import get_architecture
    from pc_variants import get_pc_variant
    torch.set_num_threads(min(2, max(1, int(os.environ.get('BENCH_CPUS', '16')))))
    torch.set_num_interop_threads(1)
    z = np.load(input_path, allow_pickle=False)
    ids, x = z['id'], torch.from_numpy(z['x']).float().div(255).sub_(0.5).div_(0.5).flatten(1)
    card = json.loads((model / 'model_card.json').read_text())
    predictions = []
    for entry in card['replicas']:
        pc = get_pc_variant('EO', True)(get_architecture('FashionMNIST', use_CELoss=True), iters=4, e_lr=.003, w_lr=5e-5)
        with np.load(model / entry['weights'], allow_pickle=False) as weights:
            pc.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights.files}, strict=True)
        pc.eval()
        values = []
        with torch.no_grad():
            for batch in x.split(64):
                pc(batch)
                values.append(pc.y_pred(batch).argmax(1).numpy())
        predictions.append(np.concatenate(values))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id'] + [f'pred{s}' for s in range(5)])
        w.writerows(zip(ids, *predictions))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--out', type=Path, required=True)
    o = sub.add_parser('train-one'); o.add_argument('--out', type=Path, required=True); o.add_argument('--slot', type=int, required=True); o.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('predict'); p.add_argument('--model', type=Path, required=True); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.command == 'train': train(a.out)
    elif a.command == 'train-one': train_one(a.out, a.slot, a.seed)
    else: predict(a.model, a.input, a.output)
