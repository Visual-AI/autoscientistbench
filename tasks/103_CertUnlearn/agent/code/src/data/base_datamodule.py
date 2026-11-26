import torch
from torch.utils.data import random_split
from jax.tree_util import tree_map
import numpy as np
from ..utils.utils import logger


def numpy_collate(batch):
    return tree_map(np.asarray, torch.utils.data.default_collate(batch))


class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        generator=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )


class BaseDataModule:
    def __init__(self, config, generator, dataset_cls, train_transform, test_transform):
        self.root = config["dataset"]["root"]
        self.batch_size = config["dataset"]["batch_size"]
        self.num_workers = config["dataset"]["num_workers"]
        self.val_split = config["dataset"]["val_split"]
        self.forget_split = config["dataset"]["forget_split"]
        self.generator = generator
        self.split_generator = torch.Generator().manual_seed(config["dataset"]["seed"])
        self.dataset_cls = dataset_cls
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self):
        from pathlib import Path
        root = Path(self.root)
        with np.load(root / "train" / "train.npz", allow_pickle=False) as f:
            self.train_dataset = ArrayDataset(f["x"], f["y"], self.train_transform)
            self.retain_dataset = torch.utils.data.Subset(self.train_dataset, f["retain"].tolist())
            self.forget_dataset = torch.utils.data.Subset(self.train_dataset, f["forget"].tolist())
        with np.load(root / "val" / "input.npz", allow_pickle=False) as x, np.load(root / "val" / "labels.npz", allow_pickle=False) as y:
            self.val_dataset = ArrayDataset(x["x"], y["y"], self.train_transform)

    def train_dataloader(self):
        return NumpyLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            generator=self.generator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def retain_dataloader(self):
        return NumpyLoader(
            self.retain_dataset,
            batch_size=self.batch_size,
            generator=self.generator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def forget_dataloader(self):
        return NumpyLoader(
            self.forget_dataset,
            batch_size=self.batch_size,
            generator=self.generator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return NumpyLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.x[i])), int(self.y[i])
