from pony import orm
from transformers import pipeline

from vkr.utils.vkr_root import VKR_ROOT

pipe = pipeline("text-generation", model="IlyaGusev/saiga_llama3_8b", device="cuda")

prompt_1 = """<|begin_of_text|><|start_header_id|> system <|end_header_id|>
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им проверять информацию, анализировать данные и отвечать на вопросы.
<|start_header_id|> user <|end_header_id|>
На основе предоставленной новости создайте поисковый запрос, чтобы проверить ее достоверность. Заголовок новости: '{news_title}'. Текст новости: '{news_body}'. Поисковый запрос должен содержать основные детали новости, чтобы в результатах не было похожих по контексту новостей. Пожалуйста, в выходные данные напиши только поисковый запрос.
<|start_header_id|> assistant <|end_header_id|>"""
prompt_2 = """<|begin_of_text|><|start_header_id|> system <|end_header_id|>
Ты — Сайга, автоматический ассистент. Определи ключевые детали новости и, затем, создай запрос для проверки её достоверности.
<|start_header_id|> user <|end_header_id|>
Пожалуйста, в выходные данные напиши только поисковый запрос. Заголовок: '{news_title}', Текст: '{news_body}'. Запрос:
<|start_header_id|> assistant <|end_header_id|>"""
prompt_3 = """<|begin_of_text|><|start_header_id|> system <|end_header_id|>
Ты — Сайга, помощник в проверке фактов. Создай поисковый запрос, избегая новостей с похожим контекстом.
<|start_header_id|> user <|end_header_id|>
Сначала определи основные вопросы, которые затрагиваются в новости. Пожалуйста, в выходные данные напиши только поисковый запрос. Заголовок: '{news_title}', Текст: '{news_body}'. Запрос:
<|start_header_id|> assistant <|end_header_id|>"""

DATABASE_NAME = 'llama3_queries.sql'
DATABASE_PATH = VKR_ROOT / 'data' / DATABASE_NAME
db_exists = DATABASE_PATH.exists()

db = orm.Database()


class QueryResult(db.Entity):
    prompt = orm.Required(str, unique=True)
    data = orm.Required(str)


db.bind(provider='sqlite', filename=DATABASE_PATH.as_posix(), create_db=True)
db.generate_mapping(create_tables=not db_exists)


@orm.db_session
def remove_search_result_by_query(prompt: str) -> None:
    result = QueryResult.get(prompt=prompt)
    if result:
        result.delete()
        orm.commit()


@orm.db_session
def insert_or_update_search_result(prompt: str, data: str) -> None:
    result = QueryResult.get(prompt=prompt)
    if result:
        result.data = data
    else:
        QueryResult(prompt=prompt, data=data)


def get_query(title: str, text: str, prompt: str, in_db: bool = False) -> str:
    result = QueryResult.get(prompt=prompt)
    if result:
        return result.data
    assert not in_db, "prepare llm queries before inference"
    news_prompt = prompt.format(news_title=title, news_body=text)
    generated_text = pipe(news_prompt, truncation=True)
    answer = generated_text[0]['generated_text'][len(news_prompt):]
    insert_or_update_search_result(prompt=prompt, data=answer)
    return answer
