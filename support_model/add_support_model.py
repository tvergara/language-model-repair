import torch
from support_model.self_attention_module import SelfAttentionModule

class AddSupportModel:
    def __init__(
        self,
        main_model,
        support_model,
        tokenization_translator,
        communicate_every_x_layers=3,
        attention_heads=3,
        key_size=20
    ):
        self.main_model = main_model
        self.support_model = support_model
        self.tokenization_translator = tokenization_translator
        combined_hidden_size = main_model.config.hidden_size + support_model.model_dim
        self.cross_attention_layers = [
            SelfAttentionModule(combined_hidden_size, attention_heads, key_size)
            for _ in range(len(self.layers()))
        ]

        for i, layer in enumerate(self.layers()):
            original_forward = layer.forward

            def modified_forward(
                *args,
                original_forward=original_forward,
                support_model=support_model,
                attn=self.cross_attention_layers[i],
                **kwargs
            ):
                output = original_forward(*args, **kwargs)
                for _ in range(communicate_every_x_layers):
                    support_output = support_model.forward_one_layer()

                tensor_output, *rest = output
                support_output = support_output.to(tensor_output.dtype)
                concat_output = torch.cat((tensor_output, support_output), dim=-1)
                modified_output = attn(concat_output)

                main_model_output, support_model_output = torch.split(
                    modified_output, [tensor_output.size(-1), support_output.size(-1)], dim=-1
                )

                return main_model_output, *rest

            layer.forward = modified_forward

    def __getattr__(self, name):
        return getattr(self.main_model, name)

    def __call__(self, *args, **kwargs):
        translated_tokens = self.tokenization_translator(kwargs['input_ids'])
        self.support_model.embed_tokens(translated_tokens)
        output = self.main_model(*args, **kwargs)
        return output

    def train_only_cross_attention(self):
        for param in self.main_model.parameters():
            param.requires_grad = False
        for param in self.support_model.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        cross_attention_params = [p for layer in self.cross_attention_layers for p in layer.parameters()]
        return cross_attention_params

    def to(self, device):
        self.main_model.to(device)
        self.support_model.to(device)
        for layer in self.cross_attention_layers:
            layer.to(device)

    def layers(self):
        model_name = type(self.main_model).__name__
        if model_name == 'GPT2LMHeadModel':
            return self.main_model.transformer.h
        else:
            raise 'Unrecognized model name'

    def generate(self, input_ids, max_length=50, do_sample=False):
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            output = self(input_ids=generated)
            logits = output.logits[:, -1, :]

            if do_sample:
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)

            if (next_token == self.main_model.config.eos_token_id).all():
                break

        return generated

