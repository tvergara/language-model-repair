import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def evaluate_tiny_shakespeare(model, tokenizer, original_model, max_length=12):
    response = requests.get(URL)
    response.raise_for_status()
    data = response.text

    lines = data.splitlines()[:2000]

    total_perplexity = 0
    total_kl_divergence = 0
    valid_lines = 0

    model.eval()

    with torch.no_grad():
        for line in tqdm(lines, desc='evaluating in tiny shakespeare'):
            if not line.strip():
                continue

            input_ids = tokenizer.encode(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            shifted_logits = logits[:, :-1, :]
            shifted_labels = input_ids[:, 1:]

            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                reduction="mean"
            )

            perplexity = torch.exp(loss).item()
            total_perplexity += perplexity

            original_outputs = original_model(input_ids=input_ids)
            original_logits = original_outputs.logits

            shifted_original_logits = original_logits[:, :-1, :]
            shifted_new_logits = shifted_logits

            original_probs = F.log_softmax(shifted_original_logits, dim=-1)
            new_probs = F.softmax(shifted_new_logits, dim=-1)

            kl_div = F.kl_div(
                original_probs.reshape(-1, original_probs.size(-1)),
                new_probs.reshape(-1, new_probs.size(-1)),
                reduction="batchmean"
            )
            total_kl_divergence += kl_div.item()

            valid_lines += 1

    if valid_lines == 0:
        return float('inf')

    avg_perplexity = total_perplexity / valid_lines
    avg_kl_divergence = total_kl_divergence / valid_lines

    return avg_perplexity, avg_kl_divergence

