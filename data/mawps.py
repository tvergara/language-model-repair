import os
import re
import random
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from torch.utils.data import Dataset, random_split

load_dotenv()

STORAGE_DIR = os.path.expanduser(os.getenv('STORAGE_DIR'))
DATASET_NAME = "mwpt5/MAWPS"

def is_addition_equation(equation, question):
    if 'N_02' in question:
        return
    return re.fullmatch(r"\S+ \+ \S+", equation)

def generate_augmented_examples(example):
    augmented_examples = []
    equation = example['Equation']

    match = re.fullmatch(r"(\S+) \+ (\S+)", equation)
    if not match:
        return []

    operand1, operand2 = match.groups()

    for _ in range(400):
        new_operand1 = str(random.randint(1, 400))
        new_operand2 = str(random.randint(1, 400))
        result = int(new_operand1) + int(new_operand2)
        new_question = example['Question'].replace(operand1, new_operand1).replace(operand2, new_operand2)
        new_question += '\nAnswer:'
        augmented_examples.append({ 'question': new_question, 'result': result })

    return augmented_examples

def get_mawps_data():
    dataset = load_dataset(
        DATASET_NAME,
        split='train',
        cache_dir=STORAGE_DIR
    )

    filtered_dataset = dataset.filter(lambda x: is_addition_equation(x['Equation'], x['Question']))

    augmented_data = []
    for example in filtered_dataset:
        augmented_data.extend(generate_augmented_examples(example))


    return augmented_data


class MAWPSDataset(Dataset):
    def __init__(self):
        self.data = get_mawps_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['question'], self.data[idx]['result']

def get_mawps():
    dataset = MAWPSDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

if __name__ == '__main__':
    train, test = get_mawps()

    m = 0
    for example in train:
        m = max(m, len(example['question']))

    m
