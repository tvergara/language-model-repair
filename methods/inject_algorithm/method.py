from compile_model.load_compiled_model import load_model

def inject_algorithm(model, tokenizer, data):
    compiled_model, compiled_tokenizer = load_model()

    return model, None
