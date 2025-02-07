import random
from random import shuffle
from collections import Counter

from torch.utils.data import Dataset, random_split

ALL_PAIRS = '()[]{}'

class DyckDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        return prompt, label

def create_random_example(size):
    counter = Counter()
    word = ''
    correct = True
    for _ in range(size):
        char = random.choice(ALL_PAIRS)
        counter[char] += 1
        word += char

        if counter['('] < counter[')'] or counter['['] < counter[']'] or counter['{'] < counter['}']:
            correct = False


    if counter['('] != counter[')'] or counter['['] != counter[']'] or counter['{'] != counter['}']:
        correct = False

    return word, correct

def create_correct_example(size):
    if size % 2 ==  1:
        size += 1
    word = ''

    counters = [0, 0, 0]
    for i in range(size):

        valid = []
        for closure, counter in zip(")]}", counters):
            if counter > 0:
                valid.append(closure)

        needs_closing = sum(counters)
        if needs_closing + i < size:
            valid += ['(', '{', '[']

        char = random.choice(valid)
        is_opening =  '([{'.find(char)
        if is_opening >= 0:
            counters[is_opening] += 1

        is_closure =  ')]}'.find(char)
        if is_closure >= 0:
            counters[is_closure] -= 1

        word += char

    return word, True

def create_dyck_data(max_length, examples):
    data = []


    for _ in range(examples // 2):
        size = random.randint(5, max_length)
        data.append(create_random_example(size))

    for _ in range(examples // 2):
        size = random.randint(5, max_length)
        data.append(create_correct_example(size))

    return data

def format_data(example):
    term, label = example
    response = 'yes' if label else 'no'
    return f"Are parenthesis here correctly matched? {term}\nAnswer:", response

def prepare_dyck_dataset(max_length=30, examples=10**6, test_size=0.2, max_test_size=4000):
    data = create_dyck_data(max_length, examples)
    shuffle(data)
    formatted_data = list(map(format_data, data))
    dataset = DyckDataset(formatted_data)

    test_size = min(int(test_size * len(dataset)), max_test_size)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


if __name__ == '__main__':
    prepare_dyck_dataset()
