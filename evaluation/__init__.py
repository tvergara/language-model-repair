from .shakespeare import evaluate_tiny_shakespeare
from .task import evaluate_task
from .sst2 import evaluate_sst2
from .lambada import evaluate_lambada
from .mrpc import evaluate_mrpc


def run_evaluations(model, tokenizer):
    results = {
        "sst2": evaluate_sst2(model, tokenizer),
        "lambada": evaluate_lambada(model, tokenizer),
        "shakespeare": evaluate_tiny_shakespeare(model, tokenizer),
        "mrpc": evaluate_mrpc(model, tokenizer),
    }

    print('Evaluation finished', results)
    return results


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from dotenv import load_dotenv
    import os
    import torch

    load_dotenv()
    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
    model_name = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model.to(device)

    run_evaluations(model, tokenizer)
