import re

import pandas as pd


def clean_text(text):
    """
    Cleans the input text specifically for use with BERT or similar models, where minimal preprocessing is needed.

    Args:
    text (str): Original text.

    Returns:
    str: Cleaned text.
    """
    # Remove HTML tags - BERT does not benefit from HTML formatting
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs - URLs are usually irrelevant to text understanding tasks
    text = re.sub(r'http\S+', '', text)
    # Remove emails - emails can usually be removed for general text understanding tasks
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove Twitter mentions - mentions don't generally add contextual meaning
    text = re.sub(r'@\w+', '', text)
    # Remove any not a letter, number, or common punctuation (keeps spaces and basic punctuation)
    text = re.sub(r'[^\w\s,.!?-]', '', text, flags=re.UNICODE)
    return ' '.join(text.split())


def welfake(df: pd.DataFrame) -> pd.DataFrame:
    print((df.label == 0).sum(), (df.label == 1).sum())
    min_title_length = 10
    min_text_length = 50

    df = df.dropna()
    df.loc[:, 'title'] = df['title'].map(clean_text)
    df.loc[:, 'text'] = df['text'].map(clean_text)
    df = df[df['title'].map(len) >= min_title_length]
    df = df[df['text'].map(len) >= min_text_length]
    df.loc[:, 'label'] = df['label'].map(lambda x: 1 - x)
    return df.reset_index(drop=True)


def fake_news_prediction(df: pd.DataFrame) -> pd.DataFrame:
    print((df.label == 'FAKE').sum(), (df.label == 'REAL').sum())
    min_title_length = 10
    min_text_length = 50

    df = df.dropna()
    df.loc[:, 'title'] = df['title'].map(clean_text)
    df.loc[:, 'text'] = df['text'].map(clean_text)
    df = df[df['title'].map(len) >= min_title_length]
    df = df[df['text'].map(len) >= min_text_length]
    df.loc[:, 'label'] = df['label'].map(lambda x: 1 if x == 'REAL' else 0)
    df['label'] = pd.to_numeric(df['label'])
    return df.reset_index(drop=True)


def russian_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    min_title_length = 10
    min_text_length = 50

    df.loc[:, 'title'] = df['title'].map(clean_text)
    df.loc[:, 'text'] = df['text'].map(clean_text)
    df = df[df['title'].map(len) >= min_title_length]
    df = df[df['text'].map(len) >= min_text_length]
    return df.reset_index(drop=True)
