from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import dill
import os
import torch
from compile_model.load_compiled_model import load_model
import torch.nn as nn

load_dotenv()

cache_dir = os.path.expanduser(os.getenv('CACHE_DIR'))
# model_name = "microsoft/phi-2"
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=cache_dir
)

model.model.layers[0]

compiled_model, compiled_tokenizer = load_model()

class AddSupportModel:
    def __init__(self, main_model, support_model, tokenization_translator, communicate_every_x_layers=3):
        self.main_model = main_model
        self.support_model = support_model
        self.tokenization_translator = tokenization_translator
        for i, layer in enumerate(self.main_model.model.layers):
            original_forward = layer.forward

            def modified_forward(*args, original_forward=original_forward, support_model=support_model, **kwargs):
                output = original_forward(*args, **kwargs)
                for _ in range(communicate_every_x_layers):
                    support_model.forward_one_layer()
                return output

            layer.forward = modified_forward

    def __getattr__(self, name):
        return getattr(self.main_model, name)

    def __call__(self, *args, **kwargs):
        self.support_model.embed_tokens(translator(kwargs['input_ids']))
        output = self.main_model(*args, **kwargs)
        return output


compiled_tokenizer.vocab

def create_translator(entry_tokenizer, output_tokenizer, important_tokens, variations=None):
    if variations is None:
        variations = [
            lambda x: x,
            lambda x: ' ' + x,
            lambda x: x + ' ',
        ]

    mapping = {}

    for token in important_tokens:
        for variation in variations:
            try:

                entry_ids = entry_tokenizer.encode(variation(token), add_special_tokens=False)

                output_ids = output_tokenizer.tokenize(token, add_special_tokens=False)


                if len(entry_ids) == 1 and len(output_ids) == 1:
                    mapping[entry_ids[0]] = output_ids[0].item()
                else:
                    pass
                    # print(f"Warning: Token '{token}' with variation '{variation(token)}' "
                    #       f"did not produce a consistent mapping.")
            except Exception as e:
                print(f"Error processing token '{token}' with variation '{variation(token)}': {e}")

    def translator(tokens):
        return torch.tensor([mapping.get(token.item(), output_tokenizer.vocab['pad']) for token in tokens.flatten()]).reshape(tokens.shape)

    return translator

important_tokens = [str(i) for i in range(10)] + ['+', '=']

translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)
compiled_model(translator(tokenizer('2 + 2 = 4', return_tensors='pt')['input_ids']))

compiled_tokenizer('2 + 2 = 4')

wrapped_model = AddSupportModel(model, compiled_model, translator)
wrapped_model(**tokenizer('2 + 2 = 4', return_tensors='pt'))
