import torch
import pandas as pd
from transformers import BertTokenizer

from vkr_prod.ml.model import NewsDataset, BertForBinaryClassification


def test_news_dataset():
    data = {'input': ['This is a test input'], 'label': [1]}
    df = pd.DataFrame(data)
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    dataset = NewsDataset(df, tokenizer)
    assert len(dataset) == 1


def test_model_predict():
    model = BertForBinaryClassification()
    input_ids = torch.tensor([[101, 102, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0, 0]])
    logits = model.forward(input_ids, attention_mask)
    assert logits.shape == (1, 1)
