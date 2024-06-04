import pathlib

from vkr_prod import __file__ as vkr_prod_file

LIB_ROOT = pathlib.Path(vkr_prod_file).resolve().parent
VKR_PROD_ROOT = (LIB_ROOT / '..' / '..').resolve()
