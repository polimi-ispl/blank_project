# Blank Project
The goal of this repository is to speed up the creation of a new project.
It contains some useful code and utilities, as well as an examlpe of code organization.


## Project organization
Folders and files should be organized as follows.

    .
    ├── bash                            # Folder containing bash scripts
    ├── data                            # Folder containing input and output data
    ├── notebooks                       # Folder containing jupyter notebooks
    ├── results                         # Folder containing results
    ├── src                             # Folder containing useful functions
    │   ├── arguments.py                # Functions to pass arguments to scripts
    │   ├── tensorflow.py               # Functions for tensorflow scipts
    │   ├── torch.py                    # Functions for pytorch scipts
    │   └── utils.py                    # Utility functions
    ├── clear_tensorboard_runs.py       # Script to clear tensorboard runs
    ├── environment_keras.yml           # Conda keras environment definition
    ├── environment_torch.yml           # Conda torch environment definition
    ├── gpu_check.py                    # Script to check the GPU setup
    ├── log_tensorboard.py              # Script to log info through tensorboard
    ├── params.py                       # Simulation parameters
    ├── pass_parameters_to_script.py    # Script to pass input arguments
    ├── train_cnn_keras.py              # Script to train a CNN with Keras
    ├── train_cnn_lightning.py          # Script to train a CNN with pytorch lightning
    └── train_cnn_pytorch.py            # Script to train a CNN with pytorch


## Getting started
To get started, prepare a python environment and check if everything runs.

### Prepare a keras enviroment
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `keras_env` environment with [*keras_environment.yml*](keras_environment.yml)
```bash
$ conda env create -f keras_environment.yml
$ conda activate keras_env
```

### Prepare a pytorch enviroment
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `torch_env` environment with [*torch_environment.yml*](torch_environment.yml)
```bash
$ conda env create -f torch_environment.yml
$ conda activate torch_env
```

### Check GPU
- Run the script [*gpu_check.py*](gpu_check.py) to check if GPUs are working correctly.


## Train a CNN
To train a CNN
- Set the desired parameters in [*params.py*](params.py) (e.g., `model_name`, `input_shape`, etc.)
- Run the script [*train_cnn_keras.py*](train_cnn_keras.py) (if using Tensorflow / Keras) or
  [*train_cnn_pytorch.py*](train_cnn_pytorch.py) (if using pytorch) or
  [*train_cnn_lightning.py*](train_cnn_lightning.py) (if using pytorch lightning)
  
The trained model and training history are saved into `trained_models_root` folder defined in *params.py*.


## Passing arguments to a python script
In many cases, it is useful to have a python script that can accept different combinations of hyperparameters.
For example, one might want to have a *train.py* that accepts as arguments `epochs` and `learning_rate`, 
in order to try different parameters values for fine-tuning the experiment.
Instead of manually set them in a file like [*params.py*](params.py), this can be done by adopting the 
[`ArgumentParser`](https://docs.python.org/3/library/argparse.html) object.
Check it out in [*pass_parameters_to_script.py*](pass_parameters_to_script.py).


## Log experiments with TensorBoard
[Tensorboard](https://www.tensorflow.org/tensorboard/) is a powerful tool for visualizing and tracking variables along
different experiments.
A common use-case is to track the losses and the metrics of different setup.

Apart from Tensorflow and PyTorch integrations, you can log pretty much everything to your tensorboard.
See our example script [*log_tensorboard.py*](log_tensorboard.py).

*Rembember*: first you have to launch the server with `tensorboard --logdir RUN_PATH`

For your convenience, you can define a bash alias to speed up the command above:
```bash
echo "alias tb='tensorboard --logdir '" >> ~/.bashrc
source ~/.bashrc
```
Now you can start the server with `tb RUN_PATH` and your python script that creates a tensorboard log to `RUN_PATH`.


## Additional resources
- [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Tensorflow / Keras](https://www.tensorflow.org/tutorials)
- [Pytorch](https://pytorch.org/tutorials/)
- [scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)
- [Tensorboard](https://www.tensorflow.org/tensorboard/get_started)
- [Pandas](https://pandas.pydata.org/docs/getting_started/index.html)

## Credits
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Paolo Bestagini
- Francesco Picetti
- Nicolò Bonettini
- Francesco Maffezzoli
