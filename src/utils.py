"""
Utilities

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
import os

import GPUtil
import tensorflow as tf


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)
    :param id: CPU or GPU device id
    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # Set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
