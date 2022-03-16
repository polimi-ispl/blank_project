"""
@Author: Francesco Picetti
@Author: Francesco Maffezzoli
"""

import os
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

import torch


def torch_on_cuda():
    return os.environ["CUDA_VISIBLE_DEVICES"] and torch.cuda.is_available()


def set_backend():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def dtype():
    if torch_on_cuda():
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor


def platform():
    if torch_on_cuda():
        # watch out! cuda for torch is 0 because it is the first torch can see! It is not the os.environ one!
        device = "cuda:0"
    else:
        device = "cpu"
    return torch.device(device)


class model_torch(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)  # 30x30
        self.pool1 = torch.nn.MaxPool2d(2, 2)   # 15x15
        self.conv2 = torch.nn.Conv2d(32, 64, 3)  # 13x13
        self.pool2 = torch.nn.MaxPool2d(2, 2)    # 6x6, no padding
        self.conv3 = torch.nn.Conv2d(64, 64, 3)  # 4x4
        self.dense1 = torch.nn.Linear(4 * 4 * 64, 64)
        self.dense2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


def load_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    state_tmp = torch.load(weights_path, map_location='cpu')
    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp
    
    incomp_keys = model.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    
    return model


def save_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loss: float, valid_loss: float,
               train_score: float, valid_score: float,
               batch_size: int, epoch: int, path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 valid_loss=valid_loss,
                 train_score=train_score,
                 valid_score=valid_score,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)
