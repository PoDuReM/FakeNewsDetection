import dataclasses

from pony import orm
import requests
import json

from vkr.data import preprocessing
from vkr.utils import json as vkr_json, net_utils, string_utils
from vkr.utils.vkr_root import VKR_ROOT
from vkr.types import Json, Link

API_KEYS = [
    '7a56d49a3e4ce2b9e1b02b9c0cdb78715b196381',
    '688e51b45144c6bcde8010035276c5fa37fa9f17',
    '78e908ddacd37bc0164a27835353ba98ab7c951a',
    'da518be9179b3b15652368ebfecbf2c0205c06e7',
]
SERPER_URL = 'https://google.serper.dev/search'

DATABASE_NAME = 'google_searches.sql'
DATABASE_PATH = VKR_ROOT / 'data' / DATABASE_NAME
db_exists = DATABASE_PATH.exists()

db = orm.Database()


class SearchResult(db.Entity):
    query = orm.Required(str, unique=True)
    data = orm.Required(str)  # Store JSON data as a string


db.bind(provider='sqlite', filename=DATABASE_PATH.as_posix(), create_db=True)
db.generate_mapping(create_tables=not db_exists)


@orm.db_session
def remove_search_result_by_query(query: str) -> None:
    result = SearchResult.get(query=query)
    if result:
        result.delete()
        orm.commit()


@orm.db_session
def insert_or_update_search_result(query: str, data: str) -> None:
    result = SearchResult.get(query=query)
    if result:
        result.data = data
    else:
        SearchResult(query=query, data=data)


def create_params(query: str, lang: str = 'ru') -> dict:
    params = {'q': query}
    if lang == 'en':
        params['location'] = 'Washington, United States'
    elif lang == 'ru':
        params['location'] = 'Moscow, Russia'
        params['gl'] = 'ru'
        params['hl'] = 'ru'
    else:
        assert False, f'Unknown language = {lang}'
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
            title=preprocessing.clean_text(
                string_utils.normalize(json_['title']).strip().strip('...')
            ),
            snippet=preprocessing.clean_text(
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
        organics.extend(search_result['organic'])
        return SearchResult(singles=[
            SingleResult.extract_from_json(s)
            for s in organics
            if 'snippet' in s or 'priceRange' in s
        ])

    def to_bert_input(self) -> str:
        return ' [SEP] '.join([s.to_bert_input() for s in self.singles])

def serper_search(
        queries: list[str],
        api_key: str | None,
        lang: str = 'ru',
        debug: bool = False,
) -> tuple[list[Json], int]:
    googled_data = [0] * len(queries)
    unknown_indices = []
    unknown_queries = []
    for i, query in enumerate(queries):
        result = SearchResult.get(query=query)
        if result:
            googled_data[i] = vkr_json.loads(result.data)
        else:
            unknown_indices.append(i)
            unknown_queries.append(create_params(query, lang=lang))
    if len(unknown_queries) == 0:
        return googled_data, 0
    if api_key is None:
        raise Exception('No API key provided to google')
    if debug:
        print('Google:')
        print(*[queries[i] for i in unknown_indices], sep='\n')

    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json',
    }
    response = requests.request(
        'POST', SERPER_URL,
        headers=headers,
        data=json.dumps(unknown_queries),
    )
    if response.status_code == 200:
        ok = True
        for i, search_result in zip(unknown_indices, response.json()):
            googled_data[i] = search_result
            if 'error' not in search_result:
                insert_or_update_search_result(queries[i], vkr_json.dumps(search_result))
            else:
                ok = False
        if ok:
            return googled_data, len(unknown_indices)
    raise Exception(f'Failed to fetch data: {response.status_code} {response.text}')
