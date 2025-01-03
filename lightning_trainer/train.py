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
    original_model=None
):
    logger = CometLogger(
        api_key= os.getenv('COMET_API_KEY'),
        project_name=os.getenv('COMET_PROJECT_NAME'),
        workspace=os.getenv('COMET_WORKSPACE'),
    )

    first_gpu = 'cuda:0'
    batch_size = 16
    max_sequence_length = 70

    hyperparams_dict = {
        'batch_size': batch_size,
        'max_sequence_length': max_sequence_length,
    }

    lightning_model = Finetuner(
        model,
        original_model,
        tokenizer=tokenizer,
        hyperparams_dict=hyperparams_dict
    )
    data_loader, val_dataloader = prepare_data_loader(
        data,
        tokenizer,
        max_length=MAX_LENGTH_INT_DATASET,
        batch_size=batch_size
    )
    natural_data_loader, natural_val_dataloader = prepare_data_loader(
        natural_data,
        tokenizer,
        max_length=max_sequence_length,
        batch_size=batch_size
    )
    unsupervised_data_loader = prepare_unsupervised_data_loader(
        unsupervised_data,
        tokenizer,
        max_length=max_sequence_length,
        batch_size=batch_size
    )
    combined_dataloaders = (data_loader, unsupervised_data_loader, natural_data_loader)
    combined_eval_dataloaders = (val_dataloader, natural_val_dataloader)
    model.to(first_gpu)
    early_stopping = EarlyStopping(
        monitor="natural_acc/dataloader_idx_1",
        patience=2,
        mode="max",
        stopping_threshold=0.9
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=7000,
        val_check_interval=100,
        # callbacks=[early_stopping]
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=combined_dataloaders,
        val_dataloaders=combined_eval_dataloaders
    )
