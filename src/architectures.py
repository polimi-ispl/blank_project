"""
CNN Architectures

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
from tensorflow.keras import layers, models
import torch.nn.functional as F
from torch import nn
import torch

# Keras architecture
def model_1(input_shape):
    """
    Simple CNN model
    :param input_shape: input data shape
    :return: CNN model
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.add(layers.Softmax())

    return model

# Torch architecture
class model_torch(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 30x30
        self.pool1 = nn.MaxPool2d(2, 2)   # 15x15
        self.conv2 = nn.Conv2d(32, 64, 3)  # 13x13
        self.pool2 = nn.MaxPool2d(2, 2)    # 6x6, no padding
        self.conv3 = nn.Conv2d(64, 64, 3)  # 4x4
        self.dense1 = nn.Linear(4*4*64, 64)
        self.dense2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


