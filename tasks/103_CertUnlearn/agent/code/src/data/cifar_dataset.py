from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
from src.data.base_datamodule import BaseDataModule
import numpy as np
import jax
from tqdm import tqdm
from torch.utils.data import random_split


def channels_last(x):
    """Picklable spelling of the released channels-last permutation."""
    return x.permute(1, 2, 0)


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, config, generator):
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(
                channels_last
            ),  # Pytorch uses channels first, but we want channels last
        ]
        train_transform = transforms.Compose(augmentations + normalize)
        test_transform = transforms.Compose(normalize)
        super().__init__(config, generator, CIFAR10, train_transform, test_transform)
