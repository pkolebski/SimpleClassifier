from typing import Tuple

import click
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EPOCH_OUTPUT
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

from simple_classifier.data import DATA_PATH
from simple_classifier.dataset import SimpleDataset


class Resnet34(pl.LightningModule):
    def __init__(self, train_dataset_path=None, val_dataset_path=None, test_dataset_path=None,
                 annotations_path=None, num_classes: int = 3, batch_size: int = 10):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = F.mse_loss(x, y)
        self.log("train/loss", loss, on_epoch=True)
        return {"loss": loss, "x": F.softmax(x), "y": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = F.mse_loss(x, y)
        self.log("validation/loss", loss, on_epoch=True)
        return {"loss": loss, "x": F.softmax(x), "y": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = F.mse_loss(x, y)
        self.log("test/loss", loss, on_epoch=True)
        return {"loss": loss, "x": F.softmax(x), "y": y}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy, f1 = self.calculate_metrics(outputs)
        self.log("train/accuracy", accuracy, on_epoch=True)
        self.log("train/f1", f1, on_epoch=True)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy, f1 = self.calculate_metrics(outputs)
        self.log("validation/accuracy", accuracy, on_epoch=True)
        self.log("validation/f1", f1, on_epoch=True)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy, f1 = self.calculate_metrics(outputs)
        self.log("test/accuracy", accuracy, on_epoch=True)
        self.log("test/f1", f1, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset:
            train_dataset = SimpleDataset(DATA_PATH / "train_set", self.annotations_path,
                                          self.train_transforms)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=4)
        return None

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        if self.val_dataset:
            val_dataset = SimpleDataset(self.val_dataset_path, self.annotations_path,
                                        self.train_transforms)
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                              num_workers=4)
        return None

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        if self.test_dataset:
            test_dataset = SimpleDataset(self.test_dataset_path, self.annotations_path,
                                         self.train_transforms)
            return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                              num_workers=4)
        return None

    @staticmethod
    def calculate_metrics(outputs) -> Tuple[float, float]:
        xs = [batch["x"] for batch in outputs]
        xs = [x.numpy() for batch in xs for x in batch]
        ys = [batch["y"] for batch in outputs]
        ys = [y.numpy() for batch in ys for y in batch]
        accuracy = accuracy_score(np.argmax(ys, axis=1), np.argmax(xs, axis=1))
        f1 = f1_score(np.argmax(ys, axis=1), np.argmax(xs, axis=1), average="macro")
        return accuracy, f1


@click.command()
@click.option("--train_dataset", default=str(DATA_PATH / "train_set"),
              help="Path to training dataset", show_default=True)
@click.option("--val_dataset", default=str(DATA_PATH / "val_set"),
              help="Path to validation dataset", show_default=True)
@click.option("--test_dataset", default=str(DATA_PATH / "train_set"),
              help="Path to test dataset", show_default=True)
@click.option("--annotations_path", default=str(DATA_PATH / "annotations.json"),
              help="Path to datasets annotations", show_default=True)
@click.option("--epochs", default=10, help="Number of epochs", show_default=True)
@click.option("--batch_size", default=64, help="Batch size", show_default=True)
def run_training(train_dataset_path, val_dataset_path, test_dataset_path, annotations_path,
                 num_classes, batch_size, max_epoch):
    model = Resnet34(train_dataset_path, val_dataset_path, test_dataset_path, annotations_path,
                     num_classes, batch_size)
    logger = loggers.TensorBoardLogger(model.__class__.__name__)
    checkpoint = ModelCheckpoint(monitor="val/loss")
    trainer = Trainer(max_epochs=max_epoch, logger=logger, log_every_n_steps=1,
                      callbacks=[checkpoint])
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__train__":
    run_training()