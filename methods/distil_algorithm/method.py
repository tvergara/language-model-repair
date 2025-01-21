from compile_model.load_compiled_model import load_model
from support_model.create_tokenizer_translator import create_translator
from support_model.add_support_model import AddSupportModel
from lightning_trainer.distil import distil
import copy

def distil_algorithm(
    model,
    tokenizer,
    data,
    unsupervised_data,
    natural_data,
    params=None,
    train_enabled=True
):
    compiled_model, compiled_tokenizer, decoder = load_model()
    important_tokens = [str(i) for i in range(10)] + ['+', '=']
    translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)

    distil(
        model,
        compiled_model,
        tokenizer,
        translator,
        data,
        unsupervised_data,
        natural_data,
        params=params
    )

    return model
