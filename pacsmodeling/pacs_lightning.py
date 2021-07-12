import os
from argparse import ArgumentParser

import pytorch_lightning as pl

import torch
import torchmetrics
import torchvision.models as models

from typing import Dict, Union

from .pacs_dataloader import (
    PACSDatasetSingleDomain,
    PACSDatasetMultipleDomain,
    PACSSamplerSingleDomainPerBatch
)

from .pacs_utils import (
    results_save_filename
)

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

        """
        Below objects need to be defined this way so that they exist
        within the LightningModule (and not within containers within the
        module - intial attempts to package them that way resulted in 
        torch tensors on incorrect devices)
        """

        # Create accuracy objects
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        # Initialize confusion matrix metrics.
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.hparam_namespace.n_classes)
        self.valid_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.hparam_namespace.n_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.hparam_namespace.n_classes)


    @staticmethod
    def add_model_specific_args(parent_parser) -> None:
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
            default=0.001
        )
        parser.add_argument("--dataloader_workers",type=int,
            default=0
        )
        parser.add_argument("--drop_last", action="store_true",
            help="Use this flag to tell the sampler to drop incomplete batches."
        )
        parser.add_argument(
            "--use_sds", action="store_true",
            help="Use this flag to activate single domain sampling."
        )

        return parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resnet18(x)
        x = self._top_layer(x)

        return torch.log_softmax(x, dim=1)

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

        # Use our custom sampler if the single domain sampling
        # flag is set.
        if self.hparam_namespace.use_sds:
            self._sds_sampler = PACSSamplerSingleDomainPerBatch(
                self._datasets["train"],
                self.hparam_namespace.batch_size,
                drop_last=self.hparam_namespace.drop_last
            )

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.nll_loss(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparam_namespace.learning_rate
        )

        steps_per_epoch = len(self._datasets["train"]) // self.hparam_namespace.batch_size

        # Correct steps per epoch for incomplete batches if not dropping them.
        if not self.hparam_namespace.drop_last:
            steps_per_epoch += 1

        print(f"Length of training dataloader: {len(self._datasets['train'])}")
        print(f"Using value of steps per epoch: {steps_per_epoch}")
        print(f"Using value max epochs: {self.trainer.max_epochs}")

        # 1000 is the default number of epochs used by pytorch-lightning.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparam_namespace.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":"step"
            }
        }

    def _get_dataloader(self, split:str) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._datasets[split], 
            batch_size = self.hparam_namespace.batch_size,
            num_workers = self.hparam_namespace.dataloader_workers
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # If using only single domain sampling for training, pass the 
        # _sds_sampler to the Dataloader as a BatchSampler. In all other 
        # cases, pass no Sampler / BatchSampler.

        if self.hparam_namespace.use_sds:
            print("Using single domain sampling...")

            return torch.utils.data.DataLoader(
                self._datasets["train"], 
                batch_sampler = self._sds_sampler,                
                num_workers = self.hparam_namespace.dataloader_workers
            )
        else:
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

        # Compute loss value and predictions
        loss = self.cross_entropy_loss(logits, y)
        predicted_probabilites = torch.exp(logits)

        # Compute accuracy with torchmetrics objects. This function requires 
        # probabilities, so we pass the exponential of the logits
        # (due to how we're using the NLL loss and log_softmax)
        #
        # PL handles calling .reset() at the end of each epoch.
        self.train_accuracy(predicted_probabilites, y)
        self.train_confusion_matrix(predicted_probabilites, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        X, y = valid_batch
        logits = self.forward(X.float())

        # Compute validation loss 
        loss = self.cross_entropy_loss(logits, y)
        predicted_probabilites = torch.exp(logits)

        # Compute validation accuracy. This function requires 
        # probabilities, so we pass the exponential of the logits
        # (due to how we're using the NLL loss and log_softmax)
        self.valid_accuracy(predicted_probabilites, y)
        self.valid_confusion_matrix(predicted_probabilites, y)

        self.log("valid_loss", loss)
        self.log('valid_acc', self.valid_accuracy, on_step=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        logits = self.forward(X.float())

        loss = self.cross_entropy_loss(logits, y)
        predicted_probabilities = torch.exp(logits)

        self.test_accuracy(predicted_probabilities, y)
        self.test_confusion_matrix(predicted_probabilities, y)

        self.log("test_loss", loss)
        self.log('test_acc', self.test_accuracy)

    def _compute_confusion_matrix(self, split):
        # Batch sampling method isn't important here
        # as we're doing exactly one pass over the data.
        # this allows us to ignore the additional logic
        # in train_dataloader 
        dataloader = self._get_dataloader(split)

        for (X, y) in dataloader:
            pred_probs = torch.exp(self.forward(X.float()))
            self.confusion_matrices[split](pred_probs, y)

        return self.confusion_matrices[split].compute()

    def compute_confusion_matrices(self, save=True) -> Dict[str, torch.Tensor]:
        cms = {
            "train": self.train_confusion_matrix.compute(),
            "val": self.valid_confusion_matrix.compute(),
            "test": self.test_confusion_matrix.compute()
        }

        if save:
            for split, cm in cms.items():
                torch.save(
                    cm, 
                    results_save_filename(self.hparam_namespace, split)
                )

        return cms
