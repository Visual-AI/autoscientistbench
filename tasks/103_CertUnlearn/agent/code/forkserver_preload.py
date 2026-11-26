"""Preload data modules while keeping the forkserver single threaded."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import torch, torchvision, numpy
import src.data.datamodule
if len(os.listdir('/proc/self/task')) != 1:
    raise RuntimeError('Data-module preload unexpectedly started native threads')
