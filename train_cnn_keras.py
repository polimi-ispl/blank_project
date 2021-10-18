"""
Train a CNN

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
import os

import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import src.tensorflow as arch
from params import trained_models_root, model_name_keras, input_shape, batch_size, epochs
from src.utils import set_gpu


# set the computation device to be used (0: first GPU, None: CPU, -1 most free GPU)...
set_gpu(id=0)
# ... and allow TensorFlow to use the memory growth (otherwise, you will occupy all the GPU memory)
arch.init_gpus()


def main():
    # Define output paths
    weights_path = os.path.join(trained_models_root, '{:s}'.format(model_name_keras), 'model')
    history_path = os.path.join(trained_models_root, '{:s}'.format(model_name_keras), 'history_keras.npy')

    # Load dataset for training and validation
    (img_tr, y_tr), (img_val, y_val) = datasets.cifar10.load_data()

    # Normalize dataset from PNG-like uint8 to floating in [0,1]
    img_tr = img_tr / 255.0
    img_val = img_val / 255.0

    # Initialize model
    model = getattr(arch, model_name_keras)(input_shape)
    model.summary()

    # Callbacks
    mod_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min')
    stop_checkpoint = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    reduce_checkpoint = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callback_list = [mod_checkpoint, stop_checkpoint, reduce_checkpoint]

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(img_tr, y_tr, batch_size=batch_size, epochs=epochs, validation_data=(img_val, y_val),
                        callbacks=callback_list)

    # Save history
    np.save(history_path, history)


if __name__ == '__main__':
    main()

