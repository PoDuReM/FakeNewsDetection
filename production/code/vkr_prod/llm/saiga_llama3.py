from transformers import pipeline

pipe = pipeline("text-generation", model="IlyaGusev/saiga_llama3_8b", device="cuda")


# Генерация текста
class SearchGenerator:
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

    def get_queries(self, title: str, text: str) -> list[str]:
        queries = []
        for prompt in [self.prompt_1, self.prompt_2, self.prompt_3]:
            news_prompt = prompt.format(news_title=title, news_body=text)
            generated_text = pipe(news_prompt, truncation=True)
            queries.append(generated_text[0]['generated_text'][len(news_prompt):])
        return queries


class OutputGenerator:
    prompt = """
<|begin_of_text|><|start_header_id|> system <|end_header_id|>
Ты — Сайга, помощник по проверке новостей.
<|start_header_id|> user <|end_header_id|>  
На основе следующей информации создайте краткий вывод о достоверности новости и предоставьте комментарий. Используй предсказанную вероятность. Пожалуйста, не добавляйте ничего лишнего в выходные данные.
Заголовок новости: '{news_title}'
Текст новости: '{news_body}'
Предсказанная вероятность достоверности: {probability}
<|start_header_id|> assistant <|end_header_id|>"""

    def __init__(self):
        pass

    def get_output(
            self,
            title: str,
            text: str,
            probability: float,
    ) -> str:
        output_prompt = self.prompt.format(
            news_title=title, news_body=text,
            probability=probability,
        )
        generated_text = pipe(output_prompt, truncation=True)
        return f"{generated_text[0]['generated_text'][len(output_prompt):]}"
