"""
@Author: Francesco Picetti
"""

import torch
import numpy as np
import gc
import os


def set_backend():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dtype():
    if os.environ["CUDA_VISIBLE_DEVICES"]:
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor
