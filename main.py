import comet_ml
import copy
import argparse
import torch
import os

from data import get_task, get_fineweb, get_mawps
from methods.get_method import get_method
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import evaluate_tiny_shakespeare, evaluate_task
from dotenv import load_dotenv
from utils import get_tokenizer, save_results

load_dotenv()
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='int-sum')
parser.add_argument('--model', type=str, default='gpt2-large', choices=['gpt2-large', 'gpt2-xl'])
parser.add_argument('--inject', type=str, default='INJECT')
parser.add_argument('--read_from_support', type=bool, default=True)
parser.add_argument('--write_to_support', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_sequence_length', type=int, default=70)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--natural_data_loss', type=bool, default=False)
args = parser.parse_args()

TASK = args.task
MODEL_NAME = args.model
DEVICE = 'cuda:0'
CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
METHOD = args.inject

train_dataset, test_dataset = get_task(TASK)
unsupervised_data = get_fineweb()
train_natural_data, test_natural_data = get_mawps()

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
model_copy = copy.deepcopy(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer = get_tokenizer()

tokenizer.pad_token = tokenizer.eos_token
method = get_method(METHOD)

model = method(model, tokenizer, train_dataset, unsupervised_data, train_natural_data, params=args)

save_id = model.save()
print('model weights saved at', save_id)

accuracy = evaluate_task(model, tokenizer, test_dataset)
perplexity, kl_divergence = evaluate_tiny_shakespeare(model, tokenizer, model_copy)

natural_accuracy = evaluate_task(model, tokenizer, test_natural_data)

print('task', TASK)
print('method', METHOD)
print('model', MODEL_NAME)
print('accuracy', accuracy)
print('perplexity', perplexity)
print('kl_divergence', kl_divergence)
print('natural_accuracy', natural_accuracy)

save_results({
    'task': TASK,
    'method': METHOD,
    'model': MODEL_NAME,
    'accuracy': accuracy,
    'perplexity': perplexity,
    'kl_divergence': kl_divergence,
    'natural_accuracy': natural_accuracy,
    'save_id': save_id
} | vars(args))
