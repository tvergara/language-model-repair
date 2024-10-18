from .memit.methods import memit, finetuning
from .in_context_learning import in_context_learning

def get_method(method):
    if method == 'NO-METHOD':
        return lambda model, tokenizer, data: (model, None)
    if method == 'ICL':
        return in_context_learning
    if method == 'MEMIT':
        return memit
    if method == 'FT':
        return finetuning

