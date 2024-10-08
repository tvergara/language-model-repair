from torch.utils.data import Dataset, random_split

MAX_INT = 100

class IntSumDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        return prompt, label

def create_int_sum_data():
    data = []
    for i in range(MAX_INT):
        for j in range(MAX_INT):
            result = i + j
            prompt = f"{i} + {j} ="
            label = str(result)
            data.append((prompt, label))
    return data

def prepare_sum_dataset(train_size=0.8):
    data = create_int_sum_data()
    dataset = IntSumDataset(data)

    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
