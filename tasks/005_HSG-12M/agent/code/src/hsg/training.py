"""Original Lightning model, loss, optimizer and validation selection metrics."""
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from hsg.gnn_baselines import get_model_instance

class LightningGNN(pl.LightningModule):
    def __init__(
        self, 
        model_name: str,
        dim_in: int,
        dim_h_gnn: int,
        dim_h_mlp: int,
        dim_out: int,
        num_layers_gnn: int,
        num_layers_mlp: int,
        dropout: float = 0.0,
        lr_init: float = 1e-3,
        lr_min: float = 1e-5,
        weight_decay: float = 0.0,
        T_0: int = 100,
        *,
        edge_dim: int | None = None,
        num_heads: int = 1,  # only for GAT | GATv2
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model_instance(
            model_name=model_name,
            dim_in=dim_in, dim_h_gnn=dim_h_gnn, 
            dim_h_mlp=dim_h_mlp, dim_out=dim_out, 
            num_layers_gnn=num_layers_gnn, 
            num_layers_mlp=num_layers_mlp,
            dropout=dropout, edge_dim=edge_dim,
            num_heads=num_heads, **model_kwargs,
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=dim_out),
            'f1': MulticlassF1Score(num_classes=dim_out, average='macro'),
        }, prefix="train_")
        self.val_metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=dim_out),
            'acc_top5': MulticlassAccuracy(num_classes=dim_out, top_k=5),
            'acc_top10': MulticlassAccuracy(num_classes=dim_out, top_k=10),
            'f1': MulticlassF1Score(num_classes=dim_out, average='macro'),
        }, prefix="val_")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr_init, amsgrad=True, weight_decay=self.hparams.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.T_0, eta_min=self.hparams.lr_min
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    def forward(self, data):
        return self.model(data)

    def _step(self, batch):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        y = batch.y.view(-1).long()
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_metrics.update(logits, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        self.val_metrics.update(logits, y)
        if not self.trainer.sanity_checking:
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

