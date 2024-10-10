import sys
import os
import contextlib

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "memit"))


from .memit.memit import apply_memit_to_model, MEMITHyperParams

def memit(model, tokenizer, data):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        return apply_memit_to_model(
            model,
            tokenizer,
            build_change_requests(data),
            get_hparams(model)
        )

def build_change_requests(data):
    requests = []
    for prompt, label in data:
        requests.append({
            "prompt": "{}",
            "subject": prompt,
            "target_new": {"str": label},
        })

    return requests

def get_hparams(model):
    params_name = (
        './methods/memit/memit/hparams/MEMIT/'
        f"{model.config._name_or_path.replace('/', '_')}.json"
    )
    hparams = MEMITHyperParams.from_json(params_name)
    return hparams


