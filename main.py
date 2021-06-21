from argparse import ArgumentParser

from dotenv import (
    load_dotenv, find_dotenv
)
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import os

from pacsmodeling import PACSLightning

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
    default="experiment"
)
parser.add_argument(
    "--random_seed", type=int,
    default=55
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
    "--test", type=bool,
    default = False
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

    # Explicit logger configuration
    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=args.experiment_name
    )

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------

    # Checkpoint model on validation loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.chkpt_dir,
        monitor="valid_loss",
        save_top_k=1,
        period=2,
        filename= args.experiment_name + "{epoch}"
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(
        log_momentum = True
    )

    # Get trainer with arguments; register callbacks
    trainer = Trainer.from_argparse_args(
        args, 
        logger=tb_logger, 
        callbacks=[checkpoint_callback, lr_monitor_callback],
        deterministic=True
    )
    
    # Build model to spec defined in arguments
    model = PACSLightning(args)

    # Train model
    trainer.fit(model)

    if args.test:
        trainer.test(
            ckpt_path="best",
            verbose=True
        )

        print(model.confusion_matrix.compute())