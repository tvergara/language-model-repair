from compile_model.transformer import Transformer
from compile_model.tokenizer import Tokenizer
from dotenv import load_dotenv
import dill
import os

load_dotenv()

def load_model(directory=os.getenv('STORAGE_DIR'), filename=os.getenv('COMPILED_MODEL')):
    jax_model = load_jax_model(directory, filename)
    decoder = build_decoder(jax_model.residual_labels)
    model, tokenizer = jax_to_torch(jax_model)

    return model, tokenizer, decoder


def load_jax_model(directory=os.getenv('STORAGE_DIR'), filename=os.getenv('COMPILED_MODEL')):
    filepath = directory + '/' + filename
    filepath = os.path.expanduser(filepath)
    print(filepath)
    with open(filepath, 'rb') as f:
        return dill.load(f)

def jax_to_torch(compiled_model):
    max_length, model_dim = compiled_model.params['pos_embed']['embeddings'].shape
    input_dim, model_dim = compiled_model.params['token_embed']['embeddings'].shape
    num_heads = compiled_model.model_config.num_heads
    num_layers = compiled_model.model_config.num_layers
    ff_dim = compiled_model.model_config.mlp_hidden_size
    seq_len = max_length
    num_classes = input_dim
    key_size = compiled_model.model_config.key_size
    model = Transformer(input_dim, model_dim, num_heads, num_layers, ff_dim, seq_len, num_classes, key_size)
    model.set_weights(compiled_model.params)

    tokenizer = Tokenizer(
        compiled_model.input_encoder.encoding_map,
        compiled_model.output_encoder.encoding_map,
        max_length
    )
    return model, tokenizer

def build_decoder(residual_space):
    parsed_dims = []
    for dim_str in residual_space:
        parts = dim_str.split(":", 1)
        if len(parts) == 2:
            name, value = parts[0].strip(), parts[1].strip()
            if name and value:
                parsed_dims.append((name, value))

    def decoder(pred):
        name_to_best = {}
        for i, (name, value) in enumerate(parsed_dims):
            score = pred[i]
            if name not in name_to_best or score > name_to_best[name][0]:
                name_to_best[name] = (score, value)

        return {name: val for name, (s, val) in name_to_best.items()}

    return decoder

if __name__ == '__main__':
    model, tokenizer, decoder = load_model()
    model.model_dim


