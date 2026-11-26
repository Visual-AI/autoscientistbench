"""Fit the original two-rank GraphSAGE protocol using materialized splits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from hsg.training import LightningGNN
from packed import IndexDataset, PackedCollator


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def make_model():
    return FastGNN(model_name='sage', dim_in=4, dim_out=24,
        dim_h_gnn=330, dim_h_mlp=128, num_layers_gnn=4, num_layers_mlp=2,
        dropout=0., lr_init=.001, lr_min=.00001, weight_decay=0., T_0=100,
        edge_dim=6, num_heads=1, kernel_size=6)


class FastGNN(LightningGNN):
    def training_step(self, batch, batch_idx):
        return self.criterion(self(batch), batch.y)


class DataModule(pl.LightningDataModule):
    def __init__(self, root):
        super().__init__()
        self.collators = {split: PackedCollator(root/split/'input.npz', root/split/'labels.csv')
                          for split in ('train', 'val')}

    def _loader(self, split):
        collate = self.collators[split]
        return DataLoader(IndexDataset(np.arange(len(collate.ids))), batch_size=6000,
                          num_workers=0, pin_memory=False, collate_fn=collate)

    def train_dataloader(self): return self._loader('train')
    def val_dataloader(self): return self._loader('val')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    seed_in = int(os.environ['BENCH_SEED'])
    seed = {1: 42, 2: 2025, 3: 666}.get(seed_in, seed_in)
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS'])); torch.set_num_interop_threads(1)
    pl.seed_everything(seed, workers=True)
    dm = DataModule(Path(os.environ['BENCH_DATA_DIR']))
    model = make_model()
    rank = int(os.environ['RANK'])

    class Protocol(pl.Callback):
        def __init__(self): self.updates = []; self.validations = []
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.updates.append({'step': trainer.global_step+1, 'epoch': trainer.current_epoch,
                'local_graphs': batch.num_graphs, 'lr': trainer.optimizers[0].param_groups[0]['lr']})
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.is_global_zero and trainer.global_step % 50 == 0:
                print(json.dumps({'step': trainer.global_step, 'epoch': trainer.current_epoch,
                                  'wall_s': time.monotonic()-start}), flush=True)
        def on_validation_end(self, trainer, module):
            if not trainer.sanity_checking:
                self.validations.append({'step': trainer.global_step, 'epoch': trainer.current_epoch,
                                         'val_f1': float(trainer.callback_metrics['val_f1'])})
        def on_train_end(self, trainer, module):
            assert trainer.global_step == 1000
            atomic_json(out/f'protocol_rank{rank}.json', {'updates': self.updates,
                'validations': self.validations, 'optimizer_steps': trainer.global_step,
                'epochs_started': len(set(u['epoch'] for u in self.updates)), 'world_size': trainer.world_size,
                'seed': seed, 'threads': torch.get_num_threads(),
                'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})

    ckpt = ModelCheckpoint(dirpath=out/'checkpoints', monitor='val_f1', mode='max',
        filename='best-{epoch:03d}-{val_f1:.6f}', save_top_k=1, save_last=False)
    trainer = pl.Trainer(accelerator='cpu', devices=2, strategy=DDPStrategy(process_group_backend='gloo'),
        max_epochs=100, max_steps=1000, accumulate_grad_batches=1, sync_batchnorm=False,
        use_distributed_sampler=True, callbacks=[ckpt, Protocol()], val_check_interval=1.0,
        check_val_every_n_epoch=1, logger=False, deterministic=False, num_sanity_val_steps=0,
        limit_train_batches=1., limit_val_batches=1., enable_progress_bar=False,
        enable_model_summary=False, enable_checkpointing=True, default_root_dir=str(out))
    trainer.fit(model, datamodule=dm)
    if trainer.is_global_zero:
        state = torch.load(ckpt.best_model_path, map_location='cpu', weights_only=False)['state_dict']
        torch.save(state, out/'weights.pt')
        parameters = {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad}
        card = {'format': 'hsg-model-v1', 'parameters': parameters,
                'buffers': {name: list(t.shape) for name, t in state.items() if name not in parameters},
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'optimizer_steps': trainer.global_step, 'epochs_started': 0,
                'seed': seed, 'bench_seed': seed_in, 'best_val_f1': float(ckpt.best_model_score),
                'weights_sha256': hashlib.sha256((out/'weights.pt').read_bytes()).hexdigest()}
        # Count epochs from the completed trace (including the final partial epoch).
        trace = json.loads((out/'protocol_rank0.json').read_text())
        card['epochs_started'] = len(set(u['epoch'] for u in trace['updates']))
        atomic_json(out/'model_card.json', card)


if __name__ == '__main__': main()
