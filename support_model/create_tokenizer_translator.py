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
        translated_tokens = torch.tensor(
            [mapping.get(token.item(), output_tokenizer.vocab['pad']) for token in tokens.flatten()]
        ).reshape(tokens.shape).to(tokens.device)

        bos_token = output_tokenizer.vocab['bos']
        bos_tensor = torch.full((translated_tokens.shape[0], 1), bos_token, dtype=torch.long, device=tokens.device)

        translated_with_bos = torch.cat((bos_tensor, translated_tokens), dim=1)
        return translated_with_bos


    return translator
