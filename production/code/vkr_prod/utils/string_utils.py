import re
import unicodedata


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


def normalize(s: str) -> str:
    return unicodedata.normalize(
        'NFC',
        unicodedata.normalize('NFKD', s)
    ).strip()
