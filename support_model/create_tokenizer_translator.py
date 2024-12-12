import torch

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
            entry_ids = entry_tokenizer.encode(variation(token), add_special_tokens=False)
            output_ids = output_tokenizer.tokenize(token, add_special_tokens=False)

            if len(entry_ids) == 1 and len(output_ids) == 1:
                mapping[entry_ids[0]] = output_ids[0].item()

    def translator(tokens):
        return torch.tensor([mapping.get(token.item(), output_tokenizer.vocab['pad']) for token in tokens.flatten()]).reshape(tokens.shape)

    return translator
