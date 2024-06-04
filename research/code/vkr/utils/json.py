import orjson
import pathlib

from vkr.types import Json


def dumps(obj: Json) -> str:
    return orjson.dumps(obj).decode('utf-8')


def loads(s: str) -> Json:
    return orjson.loads(s)


def load(path: pathlib.Path | str) -> Json:
    with open(path, "r") as fin:
        return loads(fin.read())
