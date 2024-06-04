import dataclasses

import requests
import json

from vkr_prod.utils import net_utils, string_utils
from vkr_prod.types import Json, Link

API_KEY = '22f86adc30013d388a7036f6670192d9cb2b82d0'
SERPER_URL = 'https://google.serper.dev/search'


def create_params(query: str) -> dict:
    params = {'q': query, 'location': 'Moscow, Russia'}
    return params


@dataclasses.dataclass
class SingleResult:
    url: Link
    title: str
    snippet: str

    @staticmethod
    def extract_from_json(json_: Json) -> 'SingleResult':
        snippet_key = 'snippet'
        if snippet_key not in json_:
            snippet_key = 'priceRange'
        return SingleResult(
            url=json_['link'],
            title=string_utils.clean_text(
                string_utils.normalize(json_['title']).strip().strip('...')
            ),
            snippet=string_utils.clean_text(
                string_utils.normalize(json_[snippet_key]).strip().strip('...')
            ),
        )

    def to_bert_input(self) -> str:
        return f'{self.title} [{net_utils.get_host_name(self.url)}] {self.snippet}'


@dataclasses.dataclass
class SearchResult:
    singles: list[SingleResult]

    @staticmethod
    def extract_from_json(search_result: Json) -> 'SearchResult':
        organics = []
        if 'answerBox' in search_result and 'link' in search_result['answerBox']:
            organics.append(search_result['answerBox'])
        if 'organic' in search_result:
            organics.extend(search_result['organic'])
        return SearchResult(singles=[
            SingleResult.extract_from_json(s)
            for s in organics
            if 'snippet' in s or 'priceRange' in s
        ])

    def to_bert_input(self) -> str:
        return ' [SEP] '.join([s.to_bert_input() for s in self.singles])


def search(queries: list[str]) -> list[SearchResult]:
    headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json',
    }
    response = requests.request(
        'POST', SERPER_URL,
        headers=headers,
        data=json.dumps([create_params(q) for q in queries]),
    )
    return [SearchResult.extract_from_json(x) for x in response.json()]
