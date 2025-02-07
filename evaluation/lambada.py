import json
import requests
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import os

def evaluate_lambada_last_token(model, tokenizer, url):
    """
    Evaluate GPT‑2 on Lambada by checking if the model can predict the last token.
    
    For each example, the text is tokenized using GPT‑2’s tokenizer. All tokens 
    except the last are used as context, and the model's next-token prediction is 
    compared with the actual last token.
    
    Args:
        model: The language model (e.g., GPT2LMHeadModel).
        tokenizer: The corresponding GPT‑2 tokenizer.
        url (str): URL of the Lambada JSONL file.
    
    Returns:
        The accuracy (fraction of examples for which the predicted token matches the target token).
    """
    print(f"Downloading Lambada JSONL file from: {url}")
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download fails.
    
    lines = response.text.splitlines()
    total = 0
    correct = 0

    for line in tqdm(lines, desc="Evaluating Lambada (Last Token)"):
        try:
            example = json.loads(line)
        except json.JSONDecodeError:
            continue  # Skip malformed lines

        full_text = example.get("text", "").strip()
        if not full_text:
            continue  # Skip empty examples

        # Tokenize the full text using GPT‑2’s tokenizer.
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        # Skip examples that are too short to have both context and a target token.
        if input_ids.size(1) < 2:
            continue

        # The context is all tokens except the last.
        context_ids = input_ids[:, :-1]
        # The target token is the last token.
        target_token_id = input_ids[0, -1].item()

        # Run the model on the context.
        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs.logits

        # Get the logits for the next token (i.e. after the context).
        next_token_logits = logits[0, -1, :]

        # Greedily select the token with the highest probability.
        predicted_token_id = torch.argmax(next_token_logits).item()

        if predicted_token_id == target_token_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluation on Lambada (Last Token Prediction): Accuracy = {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    from utils import get_tokenizer
    load_dotenv()
    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR', "~/.cache/huggingface"))
    save_id = '6e6efb4b-06ca-4ac6-9759-c5fd8a75d438'
    # model_name = "gpt2-large"  # Adjust the model variant as needed.
    model_name = os.path.join(CACHE_DIR, save_id)
    
    tokenizer = get_tokenizer('dyck')
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    # GPT‑2 doesn't have a pad token by default; set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # URL for the unprocessed Lambada test examples.
    lambada_url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
    
    # Evaluate GPT‑2 on Lambada using last-token prediction.
    evaluate_lambada_last_token(model, tokenizer, lambada_url)

