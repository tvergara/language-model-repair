from torch.utils.data import Dataset, random_split

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

def prepare_sum_dataset(max_int=400, test_size=0.2, max_test_size=4000):
    data = create_int_sum_data(max_int)
    dataset = IntSumDataset(data)

    test_size = min(int(test_size * len(dataset)), max_test_size)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
