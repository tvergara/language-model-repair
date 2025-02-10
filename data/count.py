import random
from random import shuffle
from collections import Counter

from torch.utils.data import Dataset, random_split

MAX_SIZE = 40

class CountDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        return prompt, label

def create_random_example(size):
    tokens = []
    counter = Counter()
    for _ in range(size):
        token = random.choice(["x", "y"])
        tokens.append(token)
        counter[token] += 1
    text = " ".join(tokens)
    return text, counter["x"]

def create_dyck_data(max_length, examples):
    seen = set()
    data = []
    while len(data) < examples:
        size = random.randint(5, max_length)
        example = create_random_example(size)
        text = example[0]
        if text in seen:
            continue
        seen.add(text)
        data.append(example)
    return data

def format_data(example):
    text, n = example
    return f"What is the number of 'X' in this text? {text}\nAnswer:", n

def prepare_count_dataset(
    max_length=MAX_SIZE,
    examples=10**6,
    test_size=0.2,
    max_test_size=4000,
):
    data = create_dyck_data(max_length, examples)
    shuffle(data)
    formatted_data = list(map(format_data, data))
    full_dataset = CountDataset(formatted_data)
    t_size = min(int(test_size * len(full_dataset)), max_test_size)
    train_size = len(full_dataset) - t_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, t_size])
    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = prepare_count_dataset()
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
