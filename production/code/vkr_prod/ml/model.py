import pathlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

from vkr_prod.utils import google, string_utils


# Model with Binary Classification Head
class BertForBinaryClassification(nn.Module):
    def __init__(self, base_model_name: str = 'DeepPavlov/rubert-base-cased') -> None:
        super(BertForBinaryClassification, self).__init__()
        self.bert = BertModel.from_pretrained(base_model_name)

        # Freeze half of the layers
        for param in self.bert.encoder.layer[:len(self.bert.encoder.layer) // 2].parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.2)  # Additional dropout layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits

    def predict(self, dataloader: DataLoader, device: str) -> np.ndarray:
        self.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                logits = self.forward(input_ids, attention_mask).squeeze(-1)
                preds = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)


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
        self.weights = np.ones(len(dataframe))

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


class NewsClassifier:
    def __init__(
            self,
            config_path: pathlib.Path,
            base_model_name: str = 'DeepPavlov/rubert-base-cased',
    ) -> None:
        self.device = 'cuda'
        self.model = BertForBinaryClassification(base_model_name).to(self.device)
        state_dict = torch.load(config_path)

        if next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)

    def preprocess(self, dataframe: pd.DataFrame, max_length: int = 512) -> DataLoader:
        dataset = NewsDataset(dataframe, tokenizer=self.tokenizer, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=4)
        return dataloader

    def predict(self, title: str, text: str, search_results: list[google.SearchResult]) -> float:
        title = string_utils.clean_text(title)
        text = string_utils.clean_text(text)
        data = [
            {'input': f'[CLS] {title} [SEP] {text} [SEP] {result.to_bert_input()}', 'label': 0}
            for result in search_results
        ]
        dataframe = pd.DataFrame(data)
        dataloader = self.preprocess(dataframe)
        predictions = self.model.predict(dataloader, self.device)
        return predictions.mean()

    @staticmethod
    def from_config(config_path: pathlib.Path | str) -> 'NewsClassifier':
        return NewsClassifier(config_path)
