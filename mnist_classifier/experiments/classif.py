import mate
from ..models.linear import Net
from ..trainers.trainer import MNISTModel
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torchvision import transforms
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch


# Init our model
mnist_model = MNISTModel(Net())

# Init DataLoader from MNIST Dataset
train_ds = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128)

# Initialize a trainer
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
if mate.is_train:
    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)
