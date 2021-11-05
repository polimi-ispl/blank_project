import copy
import json
from argparse import Namespace
from pathlib import Path
from typing import Union


def read_args(filename: Union[str, Path]) -> Namespace:
    """Read a file containing a dict of arguments, and store them in a Namespace"""
    args = Namespace()
    with open(filename, 'r') as fp:
        args.__dict__.update(json.load(fp))
    return args


def write_args(filename: Union[str, Path], args: Namespace, indent: int = 2, excluded_keys=[]) -> None:
    """
    Write a Namespace arguments to a file.
    """
    args = copy.deepcopy(args)
    if len(excluded_keys) != 0:
        for key in excluded_keys:
            if key in args:
                args.__delattr__(key)
            else:
                print(f"{key} not in args, skipping")
                pass

    with open(filename, 'w') as fp:
        json.dump(args.__dict__, fp, indent=indent)
    