import os
import pickle
from typing import Any
import urllib

from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

try:
    from IPython.display import clear_output
except:
    pass


# ====================
def get_tqdm() -> type:
    """Return tqdm.notebook.tqdm if code is being run from a notebook,
    or tqdm.tqdm otherwise"""

    if is_running_from_ipython():
        tqdm_ = notebook_tqdm
    else:
        tqdm_ = non_notebook_tqdm
    return tqdm_


# ====================
def is_running_from_ipython():
    """Determine whether or not the current script is being run from
    a notebook"""

    try:
        # Notebooks have IPython module installed
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False


# ====================
def display_or_print(obj: Any):

    if is_running_from_ipython():
        display(obj)
    else:
        print(obj)


# ====================
def mk_dir_if_does_not_exist(path):

    if not os.path.exists(path):
        os.makedirs(path)


# ====================
def save_pickle(data: Any, fp: str):
    """Save data to a .pickle file"""

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


# ====================
def load_pickle(fp: str) -> Any:
    """Load a .pickle file and return the data"""

    if 'http' in fp:
        with urllib.request.urlopen(fp) as f:
            unpickled = pickle.load(f)
    else:
        with open(fp, 'rb') as f:
            unpickled = pickle.load(f)
    return unpickled


# ====================
def try_clear_output():

    try:
        clear_output(wait=True)
    except:
        pass
