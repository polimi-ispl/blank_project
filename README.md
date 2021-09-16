# Blank Project
The goal of this repository is to speed up the creation of a new project.
It contains some useful code and utilities, as well as an examlpe of code organization.

## Project organization
Folders and files are organized as follows.

    .
    ├── notebooks                   # Folder containing notebooks
    ├── src                         # Folder containing useful functions
    │   ├── architectures.py        # CNN architectures
    │   └── utils.py                # Utility functions
    ├── environment.yml             # Conda environment definition
    ├── gpu_check.py                # Script to check if GPU setup
    ├── params.py                   # Simulation parameters
    └── train_cnn_keras.py          # Script to train a CNN with Keras

## Getting started

### Prerequisites
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `py38_tf23` environment with *environment.yml*
```bash
$ conda env create -f environment.yml
$ conda activate py38_tf23
```

### Check GPU
- Run the script *gpu_check.py* to check if GPUs are working correctly.


## Train a CNN
To train a CNN
- Set the desired parameters in *params.py* (e.g., `model_name`, `input_shape`, etc.)
- Run the script *train_cnn_keras.py* (if using Tensorflow / Keras)

The trained model and training history are saved into `trained_models_root` folder defined in *params.py*.

## Additional resources
- [Tensorflow / Keras](https://www.tensorflow.org/tutorials)
- [Pytorch](https://pytorch.org/tutorials/)
- [scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)

## Credits
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Paolo Bestagini
