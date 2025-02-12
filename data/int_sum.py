import random
from torch.utils.data import Dataset, random_split

class IntSumDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        return prompt, label

def create_int_sum_data(max_int, min_int=0):
    data = []
    for i in range(max_int):
        for j in range(max_int):
            result = i + j
            prompt = f"{i} + {j} ="
            label = str(result)
            data.append((prompt, label))
    return data

def prepare_sum_dataset(max_int=450, test_size=0.2, max_test_size=4000, length_ood=False):
    data = create_int_sum_data(max_int)
    dataset = IntSumDataset(data)
    t_size = min(int(test_size * len(dataset)), max_test_size)
    train_size = len(dataset) - t_size
    train_dataset, test_dataset = random_split(dataset, [train_size, t_size])

    if length_ood:
        data = create_int_sum_data(4500, min_int=1000)
        dataset = IntSumDataset(data)
        t_size = min(int(test_size * len(dataset)), max_test_size)
        train_size = len(dataset) - t_size
        _, test_dataset = random_split(dataset, [train_size, t_size])

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = prepare_sum_dataset(length_ood=True)

