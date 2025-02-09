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
        if (counter['('] < counter[')'] or
            counter['['] < counter[']'] or
            counter['{'] < counter['}']):
            correct = False
    if (counter['('] != counter[')'] or
        counter['['] != counter[']'] or
        counter['{'] != counter['}']):
        correct = False
    return word, correct

def create_correct_example(size):
    if size % 2 == 1:
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
        if char in "([{":
            counters["([{".index(char)] += 1
        else:
            counters[")]}".index(char)] -= 1
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

def create_random_example_new_token(size):
    """Generate a random example with the extended alphabet '()[]{}<>'."""
    extended_pairs = '()[]{}<>'
    counter = Counter()
    word = ''
    correct = True
    for _ in range(size):
        char = random.choice(extended_pairs)
        counter[char] += 1
        word += char
        if (counter['('] < counter[')'] or
            counter['['] < counter[']'] or
            counter['{'] < counter['}'] or
            counter['<'] < counter['>']):
            correct = False
    if (counter['('] != counter[')'] or
        counter['['] != counter[']'] or
        counter['{'] != counter['}'] or
        counter['<'] != counter['>']):
        correct = False
    return word, correct

def create_correct_example_new_token(size):
    """Generate a correct (balanced) example using the extended alphabet."""
    if size % 2 == 1:
        size += 1
    word = ''

    counters = [0, 0, 0, 0]
    openings = "([{<"
    closings = ")]}>"
    for i in range(size):
        valid = []

        for closure, count in zip(closings, counters):
            if count > 0:
                valid.append(closure)

        if sum(counters) + i < size:
            valid += list(openings)
        char = random.choice(valid)
        if char in openings:
            idx = openings.index(char)
            counters[idx] += 1
        else:
            idx = closings.index(char)
            counters[idx] -= 1
        word += char
    return word, True

def create_dyck_data_new_token(max_length, examples):
    data = []
    for _ in range(examples // 2):
        size = random.randint(5, max_length)
        data.append(create_random_example_new_token(size))
    for _ in range(examples // 2):
        size = random.randint(5, max_length)
        data.append(create_correct_example_new_token(size))
    return data


def create_single_type_random_example(size, paren='()'):
    counter = 0
    word = ''
    correct = True
    for _ in range(size):
        char = random.choice(paren)
        if char == paren[0]:
            counter += 1
        else:
            counter -= 1
        word += char
        if counter < 0:
            correct = False
    if counter != 0:
        correct = False
    return word, correct

def create_single_type_correct_example(size, paren='()'):
    if size % 2 == 1:
        size += 1
    word = ''
    counter = 0
    for i in range(size):
        valid = []
        if counter > 0:
            valid.append(paren[1])
        if counter + 1 <= size - i:
            valid.append(paren[0])
        char = random.choice(valid)
        if char == paren[0]:
            counter += 1
        else:
            counter -= 1
        word += char
    return word, True

def create_single_type_dyck_data(max_length, examples, paren='()'):
    data = []
    for _ in range(examples // 2):
        size = random.randint(5, max_length)
        data.append(create_single_type_random_example(size, paren))
    for _ in range(examples // 2):
        size = random.randint(5, max_length)
        data.append(create_single_type_correct_example(size, paren))
    return data


def prepare_dyck_dataset(max_length=30, examples=10**6, test_size=0.2, 
                         max_test_size=4000, ood=False, ood_new_token=False):
    data = create_dyck_data(max_length, examples)
    if ood and not ood_new_token:
        data = [ex for ex in data if not set(ex[0]).issubset(set("()"))]
    shuffle(data)
    formatted_data = list(map(format_data, data))
    full_dataset = DyckDataset(formatted_data)

    if ood_new_token:
        train_dataset = full_dataset
        ood_examples = max_test_size
        ood_data = create_dyck_data_new_token(max_length, ood_examples)
        formatted_ood_data = list(map(format_data, ood_data))
        test_dataset = DyckDataset(formatted_ood_data)
    elif not ood:
        t_size = min(int(test_size * len(full_dataset)), max_test_size)
        train_size = len(full_dataset) - t_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, t_size])
    else:
        train_dataset = full_dataset
        ood_examples = max_test_size
        ood_data = create_single_type_dyck_data(max_length, ood_examples, paren='()')
        formatted_ood_data = list(map(format_data, ood_data))
        test_dataset = DyckDataset(formatted_ood_data)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = prepare_dyck_dataset(ood_new_token=True)
