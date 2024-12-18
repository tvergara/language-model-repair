import os

import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
import lightning as L

from .finetuner import Finetuner
from .prepare_data_loader import prepare_data_loader
from .prepare_unsupervised_data_loader import prepare_unsupervised_data_loader

def train(model, tokenizer, data, unsupervised_data, original_model=None):
    logger = CometLogger(
        api_key= os.getenv('COMET_API_KEY'),
        project_name=os.getenv('COMET_PROJECT_NAME'),
        workspace=os.getenv('COMET_WORKSPACE'),
    )


    threshold = 30
    free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if torch.cuda.utilization(device_id) < threshold]
    first_gpu = free_gpus[0]

    lightning_model = Finetuner(model, original_model)
    data_loader = prepare_data_loader(data, tokenizer)
    unsupervised_data_loader = prepare_unsupervised_data_loader(unsupervised_data, tokenizer)
    combined_dataloaders = (data_loader, unsupervised_data_loader)
    model.to('cuda')
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[first_gpu],
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=2000
    )
    trainer.fit(lightning_model, combined_dataloaders)

