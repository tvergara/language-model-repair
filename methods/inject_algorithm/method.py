from compile_model.load_compiled_model import load_model
from support_model.create_tokenizer_translator import create_translator
from support_model.add_support_model import AddSupportModel
from lightning_trainer.train import train
import copy

def inject_algorithm(model, tokenizer, data, unsupervised_data):
    model_copy = copy.deepcopy(model)
    compiled_model, compiled_tokenizer = load_model()
    important_tokens = [str(i) for i in range(10)] + ['+', '=']
    translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)
    wrapped_model = AddSupportModel(model, compiled_model, translator)
    wrapped_model.train_only_cross_attention()
    train(wrapped_model, tokenizer, data, unsupervised_data, original_model=model_copy)

    return wrapped_model
