import lightning as L
import torch
import torch.nn.functional as F

class Finetuner(L.LightningModule):
    def __init__(
        self,
        model,
        original_model,
        tokenizer=None,
        lr=1e-6,
        gradient_clip_val=100.0,
        hyperparams_dict={},
        natural_data_loss=False,
        residue_loss=False,
    ):
        super().__init__()

        self.hyperparams_dict = hyperparams_dict
        self.save_hyperparameters(ignore=['original_model', 'model', 'tokenizer'])

        self.model = model
        self.original_model = original_model
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.tokenizer = tokenizer
        self.natural_data_loss = natural_data_loss
        self.residue_loss = residue_loss


    def supervised_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids, ground_truth=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, :-1].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        loss = loss * shift_loss_mask
        loss = loss.sum() / shift_loss_mask.sum()

        del input_ids, loss_mask, outputs, logits, shift_logits, shift_labels, shift_loss_mask
        torch.cuda.empty_cache()

        return loss, self.model.support_model.residual_stream_residue

    def natural_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)
        ground_truth = batch["ground_truth"].to(self.model.device)

        outputs = self.model(input_ids=input_ids, ground_truth=ground_truth)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, :-1].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        loss = loss * shift_loss_mask
        loss = loss.sum() / shift_loss_mask.sum()

        del input_ids, loss_mask, outputs, logits, shift_logits, shift_labels, shift_loss_mask
        torch.cuda.empty_cache()

        return loss, self.model.support_model.residual_stream_residue

    def unsupervised_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            outputs_original = self.original_model(input_ids=input_ids)
            logits_original = outputs_original.logits
            probs_original = F.softmax(logits_original, dim=-1)

        kl_loss = F.kl_div(probs, probs_original, reduction='batchmean', log_target=False)

        del input_ids, outputs, logits, probs, outputs_original, logits_original, probs_original
        torch.cuda.empty_cache()

        return kl_loss

    def training_step(self, batch, batch_idx):
        supervised_batch, unsupervised_batch, natural_data_batch = batch

        supervised_loss, residue_loss = self.supervised_loss(supervised_batch)
        self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=False, prog_bar=True)

        unsupervised_loss = self.unsupervised_loss(unsupervised_batch)
        self.log("unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=False, prog_bar=True)

        natural_data_loss = 0
        if self.natural_data_loss:
            natural_data_loss, natural_residue_loss = self.natural_loss(natural_data_batch)
            print('naruak_residue', natural_residue_loss)
            residue_loss += natural_residue_loss
            self.log("natural_data_loss", natural_data_loss, on_step=True, on_epoch=False, prog_bar=True)

        if not self.residue_loss:
            residue_loss = 0
        else:
            self.log("residue_loss", residue_loss, on_step=True, on_epoch=False, prog_bar=True)

        total_loss = supervised_loss + unsupervised_loss + natural_data_loss + residue_loss

        self.log("loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        acc = self.validation_accuracy(batch)
        if dataloader_idx == 0:
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("natural_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_accuracy(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids, ground_truth=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, :-1].contiguous()

        predictions = shift_logits.argmax(dim=-1)

        predictions = predictions.view(-1)
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        correct = (predictions == shift_labels) & (shift_loss_mask == 1)
        accuracy = correct.sum().float() / shift_loss_mask.sum()

        del input_ids, loss_mask, outputs, logits, shift_logits, shift_labels, shift_loss_mask, predictions
        torch.cuda.empty_cache()

        return accuracy.item()


    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.trainable_parameters(), lr=self.lr)

    def on_before_backward(self, loss: torch.Tensor):
        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.gradient_clip_val)
