"""Here we can see how to import functions and submodules of our projects."""

# this line imports all the functions of "utils"
from .utils import *
# this line imports two elements defined in "arguments"
from .arguments import read_args, write_args

# more complex: import submodules if a package is installed
from importlib.util import find_spec

if find_spec("torch"):
    from . import torch

if find_spec("tensorflow"):
    from . import tensorflow

