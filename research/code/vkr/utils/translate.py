import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from deep_translator import GoogleTranslator
import textwrap

from pony import orm

from vkr.utils import tqdm
from vkr.utils.vkr_root import VKR_ROOT

FAIL_STRING = '[[ERROR-P0 while Translation]]'

DATABASE_NAME = 'google_translates.sql'
DATABASE_PATH = VKR_ROOT / 'data' / DATABASE_NAME
db_exists = DATABASE_PATH.exists()

db = orm.Database()


class TranslateResult(db.Entity):
    query = orm.Required(str, unique=True)
    translation = orm.Required(str)


db.bind(provider='sqlite', filename=DATABASE_PATH.as_posix(), create_db=True)
db.generate_mapping(create_tables=not db_exists)


@orm.db_session
def remove_search_result_by_query(query: str) -> None:
    result = TranslateResult.get(query=query)
    if result:
        result.delete()
        orm.commit()


@orm.db_session
def insert_or_update_search_result(query: str, translation: str) -> None:
    result = TranslateResult.get(query=query)
    if result:
        result.translation = translation
    else:
        TranslateResult(query=query, translation=translation)


def split_text(text: str, max_length: int = 4999) -> list[str]:
    return textwrap.wrap(text, max_length)


# Function to translate text
def translate_text(text: str) -> str:
    try:
        result = TranslateResult.get(query=text)
        if result:
            return result.translation
        translator = GoogleTranslator(source='en', target='ru')
        chunks = split_text(text)
        translated_chunks = translator.translate_batch(chunks)
        translation = ' '.join(translated_chunks)
        insert_or_update_search_result(query=text, translation=translation)
        return translation
    except Exception as e:
        print(str(e))
        if 'too many requests' in str(e):
            raise e
        return FAIL_STRING


# Function to apply translation using multiple threads with progress bar
def apply_translation_multithreaded(
        df: pd.DataFrame, column: str, max_workers: int = 12,
) -> list[str]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_text = {executor.submit(translate_text, text): text for text in df[column]}
        for future in tqdm.tqdm(as_completed(future_to_text), total=len(future_to_text),
                                desc="Translating"):
            translated_text = future.result()
            results.append(translated_text)
    return results
