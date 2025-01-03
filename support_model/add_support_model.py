import torch
import os
import pickle
import uuid

from support_model.self_attention_module import SelfAttentionModule
from dotenv import load_dotenv

load_dotenv()
CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))

class AddSupportModel:
    def __init__(
        self,
        main_model,
        support_model,
        tokenization_translator,
        decoder,
        communicate_every_x_layers=3,
        pad_communication=0,
        attention_heads=8,
        key_size=60,
        read_from_support=True,
        write_to_support=False
    ):
        self.main_model = main_model
        self.support_model = support_model
        self.tokenization_translator = tokenization_translator
        self.decoder = decoder
        combined_hidden_size = main_model.config.hidden_size + support_model.model_dim
        n_cross_attns = len(self.layers()) - pad_communication
        self.cross_attention_layers = [
            SelfAttentionModule(combined_hidden_size, attention_heads, key_size)
            for _ in range(n_cross_attns)
        ]

        for i, layer in enumerate(self.layers()):
            cross_attn_number = i - pad_communication
            if cross_attn_number < 0 or cross_attn_number >= n_cross_attns:
                continue

            original_forward = layer.forward

            def modified_forward(
                *args,
                original_forward=original_forward,
                support_model=support_model,
                attn=self.cross_attention_layers[cross_attn_number],
                **kwargs
            ):
                output = original_forward(*args, **kwargs)
                for _ in range(communicate_every_x_layers):
                    support_output = support_model.forward_one_layer()
                    if not read_from_support:
                        support_output = torch.zeros_like(support_output)

                tensor_output, *rest = output
                support_output = support_output.to(tensor_output.dtype)

                bos_support_output, support_output = torch.split(
                    support_output, [1, support_output.size(-2) - 1], dim=-2
                )
                concat_output = torch.cat((tensor_output, support_output), dim=-1)
                modified_output = attn(concat_output)

                main_model_output, support_model_output = torch.split(
                    modified_output, [tensor_output.size(-1), support_output.size(-1)], dim=-1
                )

                if write_to_support:
                    support_model.residual_stream = support_model_output

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

    def save(self):
        unique_id = str(uuid.uuid4())
        save_path = os.path.join(CACHE_DIR, f"cross_attention_{unique_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.cross_attention_layers, f)
        return unique_id

    def load(self, unique_id):
        load_path = os.path.join(CACHE_DIR, f"cross_attention_{unique_id}.pkl")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved cross-attention layers found with ID {unique_id}")
        with open(load_path, "rb") as f:
            saved_states = pickle.load(f)
        for layer, saved_layer in zip(self.cross_attention_layers, saved_states):
            layer.load_state_dict(saved_layer.state_dict())


