import random
from torch.utils.data import Dataset, random_split


OOD_START = 6200
OOD_END = 6800

class IntSumDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        return prompt, label

def create_int_sum_data(max_int):
    data = []
    for i in range(max_int):
        for j in range(max_int):
            result = i + j
            prompt = f"{i} + {j} ="
            label = str(result)
            data.append((prompt, label))
    return data

def prepare_sum_dataset(max_int=4000, test_size=0.2, max_test_size=4000, ood=False):
    data = create_int_sum_data(max_int)

    if not ood:
        dataset = IntSumDataset(data)
        t_size = min(int(test_size * len(dataset)), max_test_size)
        train_size = len(dataset) - t_size
        train_dataset, test_dataset = random_split(dataset, [train_size, t_size])
    else:
        train_data = [example for example in data if not (OOD_START <= int(example[1]) < OOD_END)]
        test_data = [example for example in data if OOD_START <= int(example[1]) < OOD_END]

        random.shuffle(test_data)
        random.shuffle(train_data)
        test_data = test_data[:max_test_size]

        train_dataset = IntSumDataset(train_data)
        test_dataset = IntSumDataset(test_data)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = prepare_sum_dataset(ood=True)
