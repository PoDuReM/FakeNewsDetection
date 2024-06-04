from vkr_prod.llm.saiga_llama3 import SearchGenerator, OutputGenerator


def test_search_generator():
    sg = SearchGenerator()
    queries = sg.get_queries("Test Title", "Test body of the news.")
    assert len(queries) == 3
    for query in queries:
        assert isinstance(query, str)


def test_output_generator():
    og = OutputGenerator()
    output = og.get_output("Test Title", "Test body of the news.", 0.85)
    assert isinstance(output, str)
