import comet_ml
from data.get_task import get_task
from methods.get_method import get_method
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.task import evaluate_task
from evaluation.bert_score import evaluate_bert_score
from evaluation.gsm8k import evaluate_gsm8k
from dotenv import load_dotenv
from utils.modify_tokenizer import get_tokenizer

import torch
import os

load_dotenv()
torch.manual_seed(42)

TASK = 'int-sum'
# MODEL_NAME = 'EleutherAI/gpt-j-6b'
MODEL_NAME = 'gpt2-xl'
# MODEL_NAME = "google/gemma-2b"
# MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
DEVICE = 'cuda:0'
CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
METHOD = 'INJECT'

train_dataset, test_dataset = get_task(TASK)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer = get_tokenizer()

tokenizer.pad_token = tokenizer.eos_token
method = get_method(METHOD)

train_dataset

model.transformer.h
type(model).__name__

# model = method(model, tokenizer, train_dataset)[0]

accuracy = evaluate_task(model, tokenizer, test_dataset)

# evaluate_gsm8k(model, tokenizer, '')

print('task', TASK)
print('method', METHOD)
print('model', MODEL_NAME)
print('accuracy', accuracy)

# results = evaluate_bert_score(model, tokenizer, MODEL_NAME)

# print('results', results)


# model.transformer.h
