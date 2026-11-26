from src.data.cifar_dataset import CIFAR10DataModule

def get_datamodule(config, generator=None):
    return CIFAR10DataModule(config, generator)
