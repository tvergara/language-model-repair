from data.get_task import get_task
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK = 'int-sum'
MODEL_NAME = 'gpt2-xl'
DEVICE = 'cuda:1'
CACHE_DIR = '/mnt/ialabnas/homes/tvergara'

train_dataset, test_dataset = get_task(TASK)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


