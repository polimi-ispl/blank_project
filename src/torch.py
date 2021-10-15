"""
@Author: Francesco Picetti
@Author: Francesco Maffezzoli
"""

import torch
import numpy as np
import gc
import os
import torch.nn.functional as F


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


class model_torch(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)  # 30x30
        self.pool1 = torch.nn.MaxPool2d(2, 2)   # 15x15
        self.conv2 = torch.nn.Conv2d(32, 64, 3)  # 13x13
        self.pool2 = torch.nn.MaxPool2d(2, 2)    # 6x6, no padding
        self.conv3 = torch.nn.Conv2d(64, 64, 3)  # 4x4
        self.dense1 = torch.nn.Linear(4*4*64, 64)
        self.dense2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
