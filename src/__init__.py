"""Here we can see how to import functions and submodules of our projects."""

# this line imports all the functions of "utils"
from .utils import *
# this line imports two elements defined in "architectures"
from .architectures import model_1, model_torch

# more complex: import submodules if a package is installed
from importlib.util import find_spec

if find_spec("torch"):
    from . import torch

if find_spec("tensorflow"):
    from .tensorflow import init_gpus

