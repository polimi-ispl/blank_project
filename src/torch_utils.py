"""
@Author: Francesco Picetti
@Author: Francesco Maffezzoli
"""

import os

import numpy as np
import torch_utils.nn.functional as F

import torch_utils


def torch_on_cuda():
    return os.environ["CUDA_VISIBLE_DEVICES"] and torch_utils.cuda.is_available()


def set_backend():
    torch_utils.backends.cudnn.enabled = True
    torch_utils.backends.cudnn.benchmark = True
    torch_utils.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch_utils.manual_seed(seed)
    torch_utils.cuda.manual_seed(seed)


def dtype():
    if torch_on_cuda():
        return torch_utils.cuda.FloatTensor
    else:
        return torch_utils.FloatTensor


def platform():
    if torch_on_cuda():
        # watch out! cuda for torch is 0 because it is the first torch can see! It is not the os.environ one!
        device = "cuda:0"
    else:
        device = "cpu"
    return torch_utils.device(device)


class model_torch(torch_utils.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch_utils.nn.Conv2d(3, 32, 3)  # 30x30
        self.pool1 = torch_utils.nn.MaxPool2d(2, 2)   # 15x15
        self.conv2 = torch_utils.nn.Conv2d(32, 64, 3)  # 13x13
        self.pool2 = torch_utils.nn.MaxPool2d(2, 2)    # 6x6, no padding
        self.conv3 = torch_utils.nn.Conv2d(64, 64, 3)  # 4x4
        self.dense1 = torch_utils.nn.Linear(4 * 4 * 64, 64)
        self.dense2 = torch_utils.nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch_utils.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
