from .memit.methods import memit, finetuning
from .in_context_learning import in_context_learning
from .inject_algorithm.method import inject_algorithm

def get_method(method):
    if method == 'NO-METHOD':
        return lambda model, tokenizer, data, unsupervised_data: model
    if method == 'ICL':
        return in_context_learning
    if method == 'MEMIT':
        return memit
    if method == 'FT':
        return finetuning
    if method == 'INJECT':
        return inject_algorithm
    raise Exception(f"Unrecognized method {method}")


