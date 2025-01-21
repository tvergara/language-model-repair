# Language Model Repair

Can we inject a specific algorithm into an LLM without damaging its previous mechanisms? For example, most LLMs struggle with basic arithmetic like "123 + 439 =".

If we do fine-tuning on sum data + KL Divergence on unsupervised data, it works well enough. But the mechanism learned is not interpretable, and it does not really generalize to o.o.d. Is there a better way?

----

[tracr](https://github.com/google-deepmind/tracr) is a library which allows to code in a DSL called [RASP](https://arxiv.org/abs/2106.06981). Programs in `tracr` are then compilated into a _transformer model_ which implements the exact same algorithm.

Maybe we can implement algorithms in `tracr` and then destil them into an LLM. This is what this project does.


---

Index:
- `compile_model`: `tracr` code to implement sum model + helpers to load the compiled model in torch.
- `lightning_trainer`: all code to train our models using `pytorch_lightning`.
- `methods`: implementation of different methods to solve the problem (including failed previous methods).
- `support_model`: code to combine the pretrained LLM with a compiled model (a previous failed attempt).
- `data`, `evaluation` and `experiments` are self-explanatory.
