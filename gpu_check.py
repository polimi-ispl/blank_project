"""
Check GPU status

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def main():
    # Check if GPUs exist
    gpu_list = tf.config.list_physical_devices('GPU')
    print('Num GPUs Available: ', len(gpu_list))
    for gpu in gpu_list:
        print('  ', gpu)

    # Check CUDA
    print('Tensorflow is built with CUDA: ', tf.test.is_built_with_cuda())

    # Run a CNN
    print('Define a CNN model:')
    model = models.Sequential()
    model.add(layers.Dense(4, activation='sigmoid', input_shape=(8,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    print('Run a quick training:')
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.rand(2, 8), np.array([0, 1]), epochs=1)

    print('Test finished')


if __name__ == '__main__':
    main()
