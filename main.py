from data.get_task import get_task
from methods.get_method import get_method
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.task import evaluate_task
from evaluation.bert_score import evaluate_bert_score

TASK = 'int-sum'
MODEL_NAME = 'gpt2-xl'
DEVICE = 'cuda:0'
CACHE_DIR = '/mnt/ialabnas/homes/tvergara'
METHOD = 'MEMIT'

train_dataset, test_dataset = get_task(TASK)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
method = get_method(METHOD)

# model = method(model, tokenizer, train_dataset)[0]

# accuracy = evaluate_task(model, tokenizer, test_dataset)

# print(accuracy)


results = evaluate_bert_score(model, tokenizer)
