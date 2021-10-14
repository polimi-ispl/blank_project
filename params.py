"""
Parameters file

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
import os

# Folders and paths parameters
data_root = 'data'  # Root folder for data
trained_models_root = os.path.join(data_root, 'trained_models')

# CNN parameters
model_name_torch = 'model_torch'
model_name_keras = 'model_keras'
input_shape = (32, 32, 3)
batch_size = 32
epochs = 100
