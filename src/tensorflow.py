import tensorflow as tf

__all__ = [
    "init_gpus",
]

def init_gpus():
    # Set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
