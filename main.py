import os

from argparse import ArgumentParser

from dotenv import (
    load_dotenv, find_dotenv
)
from pathlib import Path
from random import randint

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pacsmodeling import (
    PACSLightning, checkpoint_save_filename, get_sds_str
)

""" 
---------------------------------------------
Front matter - handle environment variables and 
program-wide arguments.
---------------------------------------------
"""

load_dotenv(find_dotenv())

parser = ArgumentParser()

# Program-wide arguments

parser.add_argument(
    "--experiment_name", type=str,
    default="Exp"
)
parser.add_argument(
    "--random_seed", type=int,
    default=randint(0, 10000)
)
parser.add_argument(
    "--log_dir", type=Path,
    default=Path(os.getenv("LOG_DIR") or "lightning_logs")
)
parser.add_argument(
    "--chkpt_dir", type=Path,
    default=Path(os.getenv("CHKPT_DIR") or "checkpoints")
)
parser.add_argument(
    "--tensorboard_port", type=int,
    default = os.getenv("TENSORBOARD_PORT") or 6006
)
parser.add_argument(
    "--no_logging", action="store_true",
    help = "Use this flag to disable logging / Tensorboard."
)
parser.add_argument(
    "--save_cm", action="store_true",
    help= "Use this flag to save the confusion matrix."
)
parser.add_argument(
    "--test", action="store_true",
    help="Pass this flag to evaluate the model on the test set."
)
# Go get necessary arguments from the modeling module
# (See PACS_Modeling/PACS_Module.py)
parser = PACSLightning.add_model_specific_args(parser)

# Go get necessary arguments from the training module
# (see pytorch-lightning docs)
parser = Trainer.add_argparse_args(parser)

# Parse all arguments
args = parser.parse_args()

""" 
---------------------------------------------
Main script execution begins here
---------------------------------------------
"""

if __name__ == "__main__":
    # Set random seed according to argument
    pl.seed_everything(args.random_seed)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.chkpt_dir,
            monitor="valid_loss",
            save_top_k=1,
            every_n_epochs=2,
            filename=checkpoint_save_filename(args)
        )
    ]

    # Explicit logger configuration
    if args.no_logging:
        # PL takes False as an argument for the logger
        # to mean "turn off logging".
        #
        # We do not add the LR monitor callback in this
        # case.
        tb_logger=False
    else:
        # Explictly create logger; add LR monitor
        # to callbacks (LRM does not work without a 
        # logger)
        tb_logger = pl.loggers.TensorBoardLogger(
            args.log_dir,
            name=f"{args.experiment_name}-{get_sds_str(args)}-{args.random_seed}"
        )

        callbacks.append(
            pl.callbacks.LearningRateMonitor(
                log_momentum = True
            )
        )    

    # Get trainer with arguments; register callbacks
    trainer = Trainer.from_argparse_args(
        args, 
        logger=tb_logger, 
        callbacks=callbacks,
        deterministic=True
    )
    
    # Build model to spec defined in arguments
    model = PACSLightning(args)

    # Train model
    trainer.fit(model)

    model.set_test_log_strings("train")

    # Check and save best model confusion matrix for the training set.
    trainer.test(
        ckpt_path="best",
        dataloaders=model.train_dataloader(),
        verbose=True
    )

    model.zero_test_confusion_matrix(save=args.save_cm, split="train")


    model.set_test_log_strings("valid")
    trainer.test(
        ckpt_path="best",
        dataloaders=model.val_dataloader(),
        verbose=True
    )

    model.zero_test_confusion_matrix(save=args.save_cm, split="val")

    model.set_test_log_strings("test")
    
    # Save epoch-by-epoch results as confusion matrix tensors of shape
    # (epochs, n_labels, n_labels)
    if args.save_cm:
        model.save_confusion_matrix_tensors()

    if args.test: # Run the test set.
        trainer.test(
            ckpt_path="best",
            verbose=True
        )

        model.zero_test_confusion_matrix(save=args.save_cm, split="test")
        