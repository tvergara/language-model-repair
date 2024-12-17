import os

import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
import lightning as L

from .finetuner import Finetuner
from .prepare_dataloader import prepare_dataloader

def train(model, tokenizer, data):
    logger = CometLogger(
        api_key= os.getenv('COMET_API_KEY'),
        project_name=os.getenv('COMET_PROJECT_NAME'),
        workspace=os.getenv('COMET_WORKSPACE'),
    )


    threshold = 30
    free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if torch.cuda.utilization(device_id) < threshold]
    first_gpu = free_gpus[0]

    lightning_model = Finetuner(model)
    dataloader = prepare_dataloader(data, tokenizer)
    model.to('cuda')
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[first_gpu],
        enable_checkpointing=False,
        max_epochs=2
    )
    trainer.fit(lightning_model, dataloader)

