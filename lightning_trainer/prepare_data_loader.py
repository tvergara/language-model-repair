import torch
from torch.utils.data import Dataset, DataLoader

class PreTokenizedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=12):
        self.data = []
        for prompt, label in data:
            combined_text = f"{prompt} {label}"
            tokenized = tokenizer(
                combined_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            label_start_idx = len(tokenizer(prompt)['input_ids'])
            label_end_idx = len(tokenized['input_ids'][0])
            loss_mask = torch.zeros_like(tokenized['input_ids'][0])
            loss_mask[label_start_idx:label_end_idx] = 1

            self.data.append({
                "input_ids": tokenized['input_ids'].squeeze(0),
                "loss_mask": loss_mask
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_data_loader(data, tokenizer, batch_size=16, max_length=12):
    dataset = PreTokenizedDataset(data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

