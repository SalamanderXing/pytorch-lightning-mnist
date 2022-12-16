# imports pytorch lightning modules, and the regular pytorch modules
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MNISTModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.02)
