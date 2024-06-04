from vkr_prod.inference import think_about
from vkr_prod.llm.saiga_llama3 import SearchGenerator, OutputGenerator
from vkr_prod.ml.model import NewsClassifier
from vkr_prod.utils.google import SearchResult, SingleResult
from vkr_prod.utils.vkr_prod_root import VKR_PROD_ROOT


def test_think_about(mocker):
    mocker.patch('vkr_prod.utils.google.search', return_value=[
        SearchResult(singles=[SingleResult(
            url='http://example.com',
            title='Example Title',
            snippet='Example snippet.'
        )])
    ])
    search_generator = SearchGenerator()
    classifier = NewsClassifier(VKR_PROD_ROOT / 'models/ru_bert.torch')
    output_generator = OutputGenerator()

    result = think_about('Test Title', 'Test Text', search_generator, classifier, output_generator)
    assert isinstance(result, str)
