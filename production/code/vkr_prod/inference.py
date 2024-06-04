from vkr_prod.llm import saiga_llama3
from vkr_prod.ml import model
from vkr_prod.utils import google


def think_about(
        title: str,
        text: str,
        search_generator: saiga_llama3.SearchGenerator,
        classifier: model.NewsClassifier,
        output_generator: saiga_llama3.OutputGenerator,
) -> str:
    search_queries = search_generator.get_queries(title=title, text=text)
    search_results = google.search(search_queries)

    probability = classifier.predict(
        title, text,
        search_results=search_results,
    )
    output = output_generator.get_output(
        title, text,
        probability=probability,
    )
    return output
