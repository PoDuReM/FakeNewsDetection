import pathlib

from vkr import __file__ as vkr_file

LIB_ROOT = pathlib.Path(vkr_file).resolve().parent
VKR_ROOT = (LIB_ROOT / '..' / '..').resolve()
