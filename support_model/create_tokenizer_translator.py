import torch

IMPORTANT_TOKENS = {
    'int-sum': [str(i) for i in range(10)],
    'dyck': list('()[]{}'),
}
FILLUP_TOKEN = {
    'int-sum': '+',
    'dyck': ' ',
}
CUSTOM_MAPPING = {
    'int-sum': {},
    'dyck': { ':': 'compute' },
}

def create_translator(entry_tokenizer, output_tokenizer, variations=None, task='int-sum'):
    if variations is None:
        variations = [
            lambda x: x,
            lambda x: ' ' + x,
            lambda x: x + ' ',
        ]

    mapping = {}

    fillup_token = FILLUP_TOKEN[task]
    important_tokens = IMPORTANT_TOKENS[task]
    custom_mapping = CUSTOM_MAPPING[task]
    for token in important_tokens:
        for variation in variations:
            entry_ids = entry_tokenizer.encode(variation(token), add_special_tokens=False)
            output_ids = output_tokenizer.tokenize(token, add_special_tokens=False)

            if len(entry_ids) == 1 and len(output_ids) == 1:
                mapping[entry_ids[0]] = output_ids[0].item()

    for input_token, output_token in custom_mapping.items():
        for variation in variations:
            entry_ids = entry_tokenizer.encode(variation(input_token), add_special_tokens=False)
            output_ids = output_tokenizer.tokenize(output_token, add_special_tokens=False)

            if len(entry_ids) == 1 and len(output_ids) == 1:
                mapping[entry_ids[0]] = output_ids[0].item()


    def translator(tokens):
        translated_tokens = torch.tensor(
            [mapping.get(token.item(), output_tokenizer.vocab[fillup_token]) for token in tokens.flatten()]
        ).reshape(tokens.shape).to(tokens.device)

        bos_token = output_tokenizer.vocab['bos']
        bos_tensor = torch.full((translated_tokens.shape[0], 1), bos_token, dtype=torch.long, device=tokens.device)

        translated_with_bos = torch.cat((bos_tensor, translated_tokens), dim=1)
        return translated_with_bos


    return translator
