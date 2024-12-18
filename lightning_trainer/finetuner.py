import lightning as L
import torch
import torch.nn.functional as F

class Finetuner(L.LightningModule):
    def __init__(
        self,
        model,
        original_model
    ):
        super().__init__()

        self.model = model
        self.original_model = original_model
        self.lr = 1e-5

    def training_step(self, batch, batch_idx):
        supervised_batch, unsupervised_batch = batch

        # supervised batch
        input_ids = supervised_batch["input_ids"].to(self.model.device)
        loss_mask = supervised_batch["loss_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        supervised_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        masked_loss = supervised_loss * shift_loss_mask
        supervised_loss = masked_loss.sum() / shift_loss_mask.sum()
        self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=False, prog_bar=True)

        # unsupervised batch
        unsupervised_input_ids = unsupervised_batch["input_ids"].to(self.model.device)
        outputs_unsupervised = self.model(input_ids=unsupervised_input_ids)
        logits_unsupervised = outputs_unsupervised.logits
        probs_unsupervised = F.log_softmax(logits_unsupervised, dim=-1)

        with torch.no_grad():
            outputs_original = self.original_model(input_ids=unsupervised_input_ids)
            logits_original = outputs_original.logits
            probs_original = F.softmax(logits_original, dim=-1)

        kl_loss = F.kl_div(probs_unsupervised, probs_original, reduction='batchmean', log_target=False)
        self.log("unsupervised_loss", kl_loss, on_step=True, on_epoch=False, prog_bar=True)

        # total loss
        total_loss = supervised_loss + kl_loss

        self.log("loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.trainable_parameters(), lr=self.lr)
