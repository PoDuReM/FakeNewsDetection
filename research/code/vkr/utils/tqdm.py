import warnings

from tqdm.std import TqdmExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    from tqdm.autonotebook import tqdm, trange

tqdm = tqdm
trange = trange

