"""
CNN Architectures

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
from tensorflow.keras import layers, models


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
