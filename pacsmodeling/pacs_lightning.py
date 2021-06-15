import os
from argparse import ArgumentParser

from .pacs_dataloader import (
    PACSDatasetSingleDomain,
    PACSDatasetMultipleDomain
)

import pytorch_lightning as pl

import torch
import torchmetrics
import torchvision.models as models


class PACSLightning(pl.LightningModule):
    def __init__(self,
        hparam_namespace
    ):
        super().__init__()

        # Meta choices for leave-one-out domain:
        self._splits = ["train", "val", "test"]
        self.hparam_namespace = hparam_namespace

        # Bring in pretrained ResNet18, bolt on a new
        # top layer. 
        self._resnet18 = models.resnet18(pretrained=True)
        self._top_layer = torch.nn.Linear(
            in_features=1000, 
            out_features=self.hparam_namespace.n_classes
        )

        # Create accuracy objects
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        # Initialize confusion matrix metric.
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            num_classes = self.hparam_namespace.n_classes
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Takes in parser from main.py, then adds the additional
        parameters required by this module and gives back the
        parser.
        """
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False
        )

        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--domain_name", type=str,
            default="art_painting"      
        )
        parser.add_argument("--n_classes", type=int,
            default=7        
        )
        parser.add_argument("--learning_rate", type=float,
            default=0.01
        )
        parser.add_argument("--dataloader_workers",type=int,
            default=0
        )

        return parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resnet18(x)
        x = self._top_layer(x)

        return torch.log_softmax(x, dim=1)

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.nll_loss(logits, labels)

    def setup(self, stage) -> None:
        self._datasets = {}

        # Register train, val, test dataloaders. Train and
        # val contain samples from each domain except the
        # domain named in the hyperparam. Test contains those
        # examples.
        self._datasets["train"] = PACSDatasetMultipleDomain(
            self.hparam_namespace.domain_name, "train"
        )

        self._datasets["val"] = PACSDatasetMultipleDomain(
            self.hparam_namespace.domain_name, "val"
        )

        self._datasets["test"] = PACSDatasetSingleDomain(
            self.hparam_namespace.domain_name, "test"
        )

    def _get_dataloader(self, split:str) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._datasets[split], 
            batch_size = self.hparam_namespace.batch_size,
            shuffle=True,
            num_workers = self.hparam_namespace.dataloader_workers
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparam_namespace.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparam_namespace.learning_rate,
            steps_per_epoch=len(self._datasets["train"]),
            epochs=self.hparam_namespace.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._get_dataloader("val")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._get_dataloader("test")

    def training_step(self, train_batch, batch_idx):
        """
        Control of gradient accumulation (and the call of
        optimizer.zero_grad()) is handled by pytorch_lightning
        in the trainer code.
        """
        X, y = train_batch
        logits = self.forward(X.float())

        # Compute loss value
        loss = self.cross_entropy_loss(logits, y)
        
        # Compute accuracy with torchmetrics objects. This function requires 
        # probabilities, so we pass the exponential of the logits
        # (due to how we're using the NLL loss and log_softmax)
        self.train_accuracy(torch.exp(logits), y)

        self.log("train_loss", loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        X, y = valid_batch
        logits = self.forward(X.float())

        # Compute validation loss 
        loss = self.cross_entropy_loss(logits, y)

        # Compute validation accuracy. This function requires 
        # probabilities, so we pass the exponential of the logits
        # (due to how we're using the NLL loss and log_softmax)
        self.valid_accuracy(torch.exp(logits), y)

        self.log("valid_loss", loss)
        self.log('valid_acc', self.valid_accuracy, on_step=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        logits = self.forward(X.float())

        loss = self.cross_entropy_loss(logits, y)

        predicted_probabilities = torch.exp(logits)

        self.test_accuracy(predicted_probabilities, y)

        self.log("test_loss", loss)
        self.log('test_acc', self.test_accuracy)

        self.confusion_matrix(predicted_probabilities, y)
