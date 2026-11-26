"""Frozen official evaluation arithmetic, including float32 fold scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from hgr import hgr,kde
from gdp import gdp
from pairwise_fairness import pairwise_fairness
torch.set_num_threads(1)
def evaluate(y_hat,y_true,p):
    y_hat,y_true,p=(torch.tensor(np.asarray(a,dtype=np.float32)) for a in (y_hat,y_true,p))
    vals=[torch.nn.functional.l1_loss(y_true,y_hat),hgr(y_hat,p,density=kde),gdp(y_hat,p),pairwise_fairness(y_true,y_hat,p,use_label=True)]
    return torch.tensor(vals).numpy().astype(np.float32).astype(float)
