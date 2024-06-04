import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight


def calculate_weights(dataframe: pd.DataFrame, label_column: str = 'label') -> np.ndarray:
    classes = np.unique(dataframe[label_column])
    assert len(classes) == 2 and classes[0] == 0 and classes[1] == 1, f'{classes} != [0, 1]'
    class_weights = compute_class_weight('balanced', classes=classes, y=dataframe[label_column])
    weights = class_weights[dataframe[label_column].values]
    return weights


# Define the dataset
class NewsDataset(Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            tokenizer,
            add_special_tokens: bool = False,
            max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.data = dataframe
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.weights = calculate_weights(dataframe)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data.iloc[idx]
        inputs = self.tokenizer(
            item['input'], add_special_tokens=self.add_special_tokens, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors="pt",
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label']),
            'weights': torch.tensor(self.weights[idx]),
        }


def create_weighted_dataloader(dataset: NewsDataset, batch_size: int) -> DataLoader:
    sampler = WeightedRandomSampler(weights=dataset.weights,
                                    num_samples=len(dataset.weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    return dataloader


def create_standard_dataloader(dataset: NewsDataset, batch_size: int) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return dataloader
