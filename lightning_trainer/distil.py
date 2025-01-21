import os

import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
import lightning as L
import copy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from .distiler import Distiler
from .prepare_data_loader import prepare_data_loader
from .prepare_unsupervised_data_loader import prepare_unsupervised_data_loader
from .increase_learning_layer_callback import IncreaseLearningLayerCallback

MAX_LENGTH_INT_DATASET = 16
MAX_LENGTH_UNSUPERVISED = 70
BATCH_SIZE = 16

def distil(
    model,
    compiled_model,
    tokenizer,
    translator,
    data,
    unsupervised_data,
    natural_data,
    params=None,
):
    model_copy = copy.deepcopy(model)
    logger = CometLogger(
        api_key= os.getenv('COMET_API_KEY'),
        project_name='distil',
        workspace=os.getenv('COMET_WORKSPACE'),
    )

    first_gpu = 'cuda:0'
    model.to(first_gpu)
    compiled_model.to(first_gpu)
    model_copy.to(first_gpu)

    lightning_model = Distiler(
        model,
        compiled_model,
        model_copy,
        translator,
    )
    data_loader, val_dataloader = prepare_data_loader(
        data,
        tokenizer,
        max_length=MAX_LENGTH_INT_DATASET,
        batch_size=BATCH_SIZE
    )
    unsupervised_data_loader = prepare_unsupervised_data_loader(
        unsupervised_data,
        tokenizer,
        max_length=MAX_LENGTH_UNSUPERVISED,
        batch_size=BATCH_SIZE
    )
    natural_data_loader, natural_val_dataloader = prepare_data_loader(
        natural_data,
        tokenizer,
        max_length=MAX_LENGTH_UNSUPERVISED,
        batch_size=params.batch_size
    )
    combined_dataloaders = (data_loader, unsupervised_data_loader, natural_data_loader)

    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=params.train_batches,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=combined_dataloaders,
    )
