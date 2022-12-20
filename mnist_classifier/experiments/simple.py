import mate
from ..models.linear import Net
from ..trainers.trainer import MNISTModel
from ..data_loaders.mnist import MNISTDataModule
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch


# Init our model
mnist_model = MNISTModel(Net())

data_module = MNISTDataModule()

# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=1,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
if mate.is_train:
    # Train the model ⚡
    trainer.fit(mnist_model, data_module)
    mate.result(trainer.logged_metrics)
