"""Target routines extracted verbatim from the reproduced official notebook driver.
Only selection-loader naming is changed. See provenance in the task README.
"""
import copy, math, os, random, time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
SEED=0
device='cpu'
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    name: str
    dataset: str
    num_classes: int
    input_size: int
    batch_size: int = 128
    epochs: int = 200
    optimizer: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    cosine: bool = True
    warmup_epochs: int = 5
    cifar_style_resnet: bool = False


class FCNet(nn.Module):
    def __init__(self, in_dim: int, widths):
        super().__init__()
        dims = [in_dim] + list(widths)
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion=None):
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, 100.0 * total_correct / n


def _make_optim_sched(model: nn.Module, cfg: TrainCfg):
    if cfg.optimizer.upper() == "ADAM":
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sch = ConstantLR(opt, factor=1.0, total_iters=cfg.epochs)
        return opt, sch
    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.cosine:
        warm = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=max(1, cfg.warmup_epochs))
        cos  = CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs))
        sch  = SequentialLR(opt, schedulers=[warm, cos], milestones=[cfg.warmup_epochs])
    else:
        sch = ConstantLR(opt, factor=1.0, total_iters=cfg.epochs)
    return opt, sch


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          cfg: TrainCfg, tag: str, ckpt_dir: str = "checkpoints") -> float:
    os.makedirs(ckpt_dir, exist_ok=True)
    ce = nn.CrossEntropyLoss()
    opt, sch = _make_optim_sched(model, cfg)
    best = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
        _, val_acc = evaluate(model, val_loader)
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                       os.path.join(ckpt_dir, f"{tag}_best.pth"))
        print(f"  [{epoch+1:03d}/{cfg.epochs}] val_acc={val_acc:.2f}% best={best:.2f}% "
              f"lr={opt.param_groups[0]['lr']:.5f} t={time.time()-t0:.1f}s", flush=True)
    return best


def EMP_global_magnitude(model: nn.Module, beta: float = 1.0) -> float:
    """EMP global magnitude pruning, in place on ``model``; returns the global sparsity.

    Verbatim behaviour of scripts/experiments.ipynb cell 6 (the deep copy is used only
    to build the score vector; the mask is applied to ``model`` itself)."""
    model1 = copy.deepcopy(model)
    params = []
    for m in model1.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params.append(m.weight.data.view(-1))
    if not params:
        return 0.0

    s = torch.cat(params)
    x_norm = torch.abs(s) / torch.sum(torch.abs(s))
    neff = 1 / torch.sum((x_norm ** 2))
    r_neff = torch.floor(beta * neff)
    r_neff = r_neff.clamp(min=1, max=len(s) - 1)
    _, indices = torch.sort(torch.abs(s), descending=True)
    thresh = torch.abs(s)[indices[int(r_neff)]]

    total, kept = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            mask = (torch.abs(w) >= thresh)
            kept += mask.sum().item()
            total += mask.numel()
            w.mul_(mask)

    del model1
    return 1.0 - (kept / total)


CFG=TrainCfg("FC12", "CIFAR10", 10, input_size=32, lr=0.01, cosine=False, warmup_epochs=0, weight_decay=0.0)
