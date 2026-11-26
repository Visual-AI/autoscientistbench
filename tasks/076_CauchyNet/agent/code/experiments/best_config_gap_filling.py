"""Official model and training/scoring routines; dataset-specific material omitted."""
import json, math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ReciprocalActivation(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps
    def forward(self, x): return 1.0/(x + self.eps)

class CauchyNet(nn.Module):
    """Best-config CauchyNet from the grand sweep.

    Fixed elliptical poles (r_re, r_im); only lambda_ is trained.
    Default (r_re, r_im) = (2.5, 0.4) is OPTIMAL for inputs x in [-2, 2].
    For other input ranges, choose (r_re, r_im) so the ellipse strictly
    encloses the input domain while staying close to the real axis
    (small r_im).
    """
    def __init__(self, input_size=1, hidden_size=128, output_size=1,
                 r_re=2.5, r_im=0.4, seed=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_ = nn.Parameter(
            torch.randn(hidden_size, output_size, dtype=torch.cfloat))
        g = torch.Generator(); g.manual_seed(seed)
        angles = 2*math.pi*torch.rand(hidden_size, generator=g)
        real_part = r_re*torch.cos(angles)
        imag_part = r_im*torch.sin(angles)
        self.register_buffer(
            "xi_fixed", torch.complex(real_part, imag_part))
        self.activation = ReciprocalActivation()

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B)
        x_c = torch.complex(x_flat, torch.zeros_like(x_flat))
        x_e = x_c.unsqueeze(1).expand(B, self.hidden_size)
        xi_e = self.xi_fixed.unsqueeze(0).expand(B, self.hidden_size)
        activated = self.activation(xi_e - x_e)
        out_c = torch.matmul(activated, self.lambda_) / self.hidden_size
        return out_c.real, out_c.imag

def train_one(name, model, train_loader, val_loader,
                    epochs=2000, lr=0.01, imag_pen=0.0,
                    sched_step=2000, sched_gamma=0.5, log=False):
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=sched_gamma)
    crit = nn.MSELoss()
    t0 = time.time()
    best_val, best_state = float('inf'), None
    best_epoch = None
    train_curve, val_curve = [], []
    for ep in range(epochs):
        model.train(True)
        epoch_train_loss, n_train = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                yr, yi = out
                loss = crit(yr, yb) + imag_pen*crit(yi, torch.zeros_like(yi))
            else:
                loss = crit(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_train_loss += loss.item(); n_train += 1
        sch.step()
        train_curve.append(epoch_train_loss / max(n_train, 1))
        model.train(False)
        with torch.no_grad():
            vl, n = 0., 0
            for xb, yb in val_loader:
                out = model(xb)
                if isinstance(out, tuple): out = out[0]
                vl += crit(out, yb).item(); n += 1
            vl /= n
            val_curve.append(vl)
            if vl < best_val:
                best_val = vl
                best_epoch = ep + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    if best_state is not None: model.load_state_dict(best_state)

    return {"state_dict": model.state_dict(), "name": name,
            "epochs_completed": epochs, "best_epoch": best_epoch,
            "best_validation_mse": best_val, "lr": lr,
            "train_curve": train_curve, "val_curve": val_curve,
            "train_time_s": train_time}
