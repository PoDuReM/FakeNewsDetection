import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel

from vkr.utils import tqdm


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
            for batch in tqdm.tqdm(dataloader, desc='predict'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                logits = self.forward(input_ids, attention_mask).squeeze(-1)
                preds = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)
