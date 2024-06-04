from vkr_prod.utils.google import create_params, SingleResult, SearchResult, search


def test_create_params():
    params = create_params("test query")
    assert isinstance(params, dict)
    assert params['q'] == "test query"


def test_single_result():
    json_data = {
        'link': 'http://example.com',
        'title': 'Example Title',
        'snippet': 'This is a snippet.'
    }
    result = SingleResult.extract_from_json(json_data)
    assert result.url == 'http://example.com'
    assert result.title == 'Example Title'
    assert result.snippet == 'This is a snippet'
