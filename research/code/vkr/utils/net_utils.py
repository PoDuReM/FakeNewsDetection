import urllib.parse

from vkr.types import Link


def get_host_name(url: Link) -> str:
    return urllib.parse.urlparse(url).hostname
