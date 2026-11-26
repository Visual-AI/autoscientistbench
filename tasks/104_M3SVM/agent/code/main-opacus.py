"""PMSVM-AGP: original loss and Adam loop, materialized data protocol."""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def Regularized_loss(model, n, y_pred, y_true, p=4, lam=0.01):
    y_true = torch.argmax(y_true, dim=1)
    classification_loss = F.multi_margin_loss(y_pred, y_true) # hinge loss
    # classification_loss = -torch.mean(y_true * torch.log_softmax(y_pred, dim=1)) # logits
    
    # Fix for accessing weights in GradSampleModule
    if hasattr(model, "_module"):
        weight = model._module.weight
    else:
        weight = model.weight
        
    RG_loss = 1/n * torch.norm(weight.unsqueeze(1) - weight.unsqueeze(0), p=2, dim=2).pow(p).sum()
    loss = classification_loss + lam*RG_loss
    return loss


def fit(X_train, y_train, rng_seed, normalizer=1560):
    # Preserve the authors' full-dataset regularizer denominator.
    torch.manual_seed(int(rng_seed))
    np.random.seed(int(rng_seed))
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    n, d, num_class = normalizer, X_train.shape[1], 26
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = torch.nn.Linear(d, num_class)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)
    loss_fn = Regularized_loss
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        epochs=30, target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1,
    )
    for epoch in range(30):
        total_loss = 0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            y_batch_one_hot = F.one_hot(y_batch, num_classes=num_class).float()
            y_pred = model(X_batch)
            loss = loss_fn(model, n, y_pred, y_batch_one_hot, 2, 0.001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
    state = model._module.state_dict()
    return dict(weight=state['weight'].tolist(), bias=state['bias'].tolist(),
                history=privacy_engine.accountant.history, delta=1e-5,
                epsilon=privacy_engine.get_epsilon(delta=1e-5),
                train_samples=len(X_train), batch_size=128, clip=1,
                steps=sum(row[2] for row in privacy_engine.accountant.history),
                rng_seed=int(rng_seed))
