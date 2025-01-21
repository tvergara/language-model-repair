import os

import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from .finetuner import Finetuner
from .prepare_data_loader import prepare_data_loader
from .prepare_unsupervised_data_loader import prepare_unsupervised_data_loader

MAX_LENGTH_INT_DATASET = 16

def train(
    model,
    tokenizer,
    data,
    unsupervised_data,
    natural_data,
    original_model=None,
    params=None
):
    logger = CometLogger(
        api_key= os.getenv('COMET_API_KEY'),
        project_name=os.getenv('COMET_PROJECT_NAME'),
        workspace=os.getenv('COMET_WORKSPACE'),
    )

    first_gpu = 'cuda:0'

    lightning_model = Finetuner(
        model,
        original_model,
        tokenizer=tokenizer,
        lr=params.lr,
        natural_data_loss=params.natural_data_loss,
        residue_loss=params.residue_loss,
        hyperparams_dict=vars(params),
    )
    data_loader, val_dataloader = prepare_data_loader(
        data,
        tokenizer,
        max_length=MAX_LENGTH_INT_DATASET,
        batch_size=params.batch_size
    )
    natural_data_loader, natural_val_dataloader = prepare_data_loader(
        natural_data,
        tokenizer,
        max_length=params.max_sequence_length,
        batch_size=params.batch_size
    )
    unsupervised_data_loader = prepare_unsupervised_data_loader(
        unsupervised_data,
        tokenizer,
        max_length=params.max_sequence_length,
        batch_size=params.batch_size
    )
    combined_dataloaders = (data_loader, unsupervised_data_loader, natural_data_loader)
    combined_eval_dataloaders = (val_dataloader, natural_val_dataloader)
    model.to(first_gpu)
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=params.train_batches,
        val_check_interval=100,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=combined_dataloaders,
        val_dataloaders=combined_eval_dataloaders
    )
