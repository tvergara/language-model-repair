from transformers import GPT2TokenizerFast
import os
import requests

def get_tokenizer():
    base_url = "https://huggingface.co/gpt2/resolve/main/"
    vocab_file = "vocab.json"
    merges_file = "merges.txt"

    if not os.path.exists(vocab_file):
        vocab_url = base_url + vocab_file
        r = requests.get(vocab_url)
        r.raise_for_status()
        with open(vocab_file, "wb") as f:
            f.write(r.content)

    if not os.path.exists(merges_file):
        merges_url = base_url + merges_file
        r = requests.get(merges_url)
        r.raise_for_status()
        with open(merges_file, "wb") as f:
            f.write(r.content)

    def forms_multi_digit_number(t1, t2):
        combined = t1 + t2
        combined = combined.replace("Ġ", " ")
        combined_stripped = combined.strip()
        return combined_stripped.isdigit() and len(combined_stripped) > 1

    with open(merges_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header = []
    merge_rules = []
    for line in lines:
        if line.startswith('#'):
            header.append(line)
        else:
            merge_rules.append(line)

    filtered_merge_rules = []
    for rule in merge_rules:
        pair = rule.split()
        if len(pair) == 2:
            t1, t2 = pair
            if forms_multi_digit_number(t1, t2):
                continue
            filtered_merge_rules.append(rule)

    modified_merges_file = "modified_merges.txt"
    with open(modified_merges_file, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for line in filtered_merge_rules:
            f.write(line + "\n")

    modified_tokenizer = GPT2TokenizerFast(vocab_file=vocab_file, merges_file=modified_merges_file)
    return modified_tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer()
    tokenizer('1+1=2')
    tokenizer('1 + 1 = 2')

    from compile_model.load_compiled_model import load_model
    compiled_model, compiled_tokenizer = load_model()
    from support_model.create_tokenizer_translator import create_translator
    important_tokens = [str(i) for i in range(10)] + ['+', '=']
    translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)

    tokenizer('1 + 1 = 2')
    translator(tokenizer('1 + 1 = 2', )['input_ids'])
