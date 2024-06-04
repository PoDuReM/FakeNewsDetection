import unicodedata


def normalize(s: str) -> str:
    return unicodedata.normalize(
        'NFC',
        unicodedata.normalize('NFKD', s)
    ).strip()
