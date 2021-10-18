"""
Utilities

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
import os
import GPUtil

__all__ = [
    "set_gpu",
]


def set_gpu(id=-1):
    """
    Set tensor computation device.
    
    :param id: CPU or GPU device id (None for CPU, -1 for the device with lowest memory usage, or the ID)

    hint: use gpustat (pip install gpustat) in a bash CLI, or gputil (pip install gputil) in python.

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
        