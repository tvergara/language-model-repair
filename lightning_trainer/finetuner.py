import lightning as L
import torch
import torch.nn.functional as F

class Finetuner(L.LightningModule):
    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.model = model
        self.lr = 1e-5

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)


        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        masked_loss = loss * shift_loss_mask
        loss = masked_loss.sum() / shift_loss_mask.sum()

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.trainable_parameters(), lr=self.lr)
