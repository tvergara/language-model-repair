import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiler(L.LightningModule):
    def __init__(
        self,
        model,
        compiled_model,
        original_model,
        translator,
    ):
        super().__init__()

        self.model = model
        self.compiled_model = compiled_model
        self.original_model = original_model
        self.tokenizer_translator = translator
        self.learnable_layers = len(compiled_model.layers) + 1
        self.adapter = nn.Linear(model.config.hidden_size, compiled_model.model_dim)
        self.lr=1e-6
        self.adapter_lr=1e-4
        self.current_learning_layer = self.learnable_layers - 1
        self.adapter.to(self.model.device)


    def training_step(self, batch, batch_idx):
        supervised_batch, unsupervised_batch, natural_batch = batch

        unsupervised_loss = self.unsupervised_loss(unsupervised_batch)
        self.log("unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=False, prog_bar=True)

        algorithm_loss = self.algorithm_loss(supervised_batch)
        self.log("algorithm_loss", algorithm_loss, on_step=True, on_epoch=False, prog_bar=True)

        natural_loss = self.algorithm_loss(natural_batch)
        self.log("natural_loss", natural_loss, on_step=True, on_epoch=False, prog_bar=True)

        loss = unsupervised_loss + algorithm_loss + natural_loss
        return loss

    def algorithm_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        translated_tokens = self.tokenizer_translator(batch['input_ids'])
        self.compiled_model.embed_tokens(translated_tokens)

        loss = 0
        for i in range(self.current_learning_layer + 1):
            adaptation = self.adapter(outputs.hidden_states[i])

            tensor1 = F.normalize(adaptation, p=2, dim=-1)
            tensor2 = F.normalize(self.compiled_model.residual_stream[:, 1:], p=2, dim=-1)
            cosine_similarity = torch.sum(tensor1 * tensor2, dim=-1)
            cosine_distance = 1 - cosine_similarity
            loss += cosine_distance.mean()
            self.compiled_model.forward_one_layer()
        return loss

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

        return kl_loss


    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.lr},
                {"params": self.adapter.parameters(), "lr": self.adapter_lr},
            ]
        )
