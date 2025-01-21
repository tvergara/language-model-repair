import comet_ml
import copy
import argparse
import torch
import pickle
import os
import uuid

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
parser.add_argument('--inject', type=str, default='DISTIL')
parser.add_argument('--read_from_support', type=lambda x: str(x).lower() == 'true', default=True)
parser.add_argument('--write_to_support', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--tanh_in_write', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_sequence_length', type=int, default=70)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--natural_data_loss', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--residue_loss', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--train_batches', type=int, default=4000)
parser.add_argument('--cross_attn_every_x_layers', type=int, default=1)
parser.add_argument('--attention_heads', type=int, default=8)
parser.add_argument('--key_size', type=int, default=60)
parser.add_argument('--rescaling_factor_write', type=int, default=1)
parser.add_argument('--pad_communication', type=int, default=0)
args = parser.parse_args()

TASK = args.task
MODEL_NAME = args.model
CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
METHOD = args.inject

train_dataset, test_dataset = get_task(TASK)
unsupervised_data = get_fineweb()
train_natural_data, test_natural_data = get_mawps()

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model_copy = copy.deepcopy(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer = get_tokenizer()

tokenizer.pad_token = tokenizer.eos_token
method = get_method(METHOD)

model = method(model, tokenizer, train_dataset, unsupervised_data, train_natural_data, params=args)


save_id = str(uuid.uuid4())
save_path = os.path.join(CACHE_DIR, save_id)
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
print('model weights saved at', save_id)
