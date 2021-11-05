# Blank Project
The goal of this repository is to speed up the creation of a new project.
It contains some useful code and utilities, as well as an examlpe of code organization.


## Project organization
Folders and files should be organized as follows.

    .
    ├── bash                            # Folder containing bash scripts
    ├── data                            # Folder containing input and output data
    ├── envs                            # Folder containing conda environment definitions
    ├── notebooks                       # Folder containing jupyter notebooks
    ├── results                         # Folder containing results
    ├── src                             # Folder containing useful functions
    │   ├── arguments.py                # Functions to pass arguments to scripts
    │   ├── tensorflow_utils.py         # Functions for tensorflow scipts
    │   ├── torch_utils.py              # Functions for pytorch scipts
    │   └── utils.py                    # Utility functions
    ├── clear_tensorboard_runs.py       # Script to clear tensorboard runs
    ├── gpu_check.py                    # Script to check the GPU setup
    ├── log_tensorboard.py              # Script to log info through tensorboard
    ├── params.py                       # Simulation parameters
    ├── pass_parameters_to_script.py    # Script to pass input arguments
    ├── run_parallel.py                 # Script to run a function in parallel
    ├── train_cnn_keras.py              # Script to train a CNN with Keras
    ├── train_cnn_lightning.py          # Script to train a CNN with pytorch lightning
    └── train_cnn_pytorch.py            # Script to train a CNN with pytorch


## Getting started
To get started, prepare a python environment and check if everything runs.

### Prepare a keras enviroment
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `keras_env` environment with [*keras_environment.yml*](keras_environment.yml)
```bash
$ conda env create -f envs/keras_environment.yml
$ conda activate keras_env
```

### Prepare a pytorch enviroment
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `torch_env` environment with [*torch_environment.yml*](torch_environment.yml)
```bash
$ conda env create -f envs/torch_environment.yml
$ conda activate torch_env
```

### Use a GPU
Considering the high computational cost of deep learning algorithms, it is useful to accelarate on a GPU.
Basically, both tensorflow and pytorch come with native GPU support. However, they have different features:
- PyTorch natively supports 1 GPU at a time; if you (really) need multiple GPUs, you have to code your own 
`DataParallel` paradigm. To the best of our knowledge, it works perfectly at a glance.
- Tensorflow typically needs an extra effort, as it needs a [perfect combination](https://www.tensorflow.org/install/source#gpu) of driver/python/package versions. Run the script [*gpu_check_tf.py*](gpu_check_tf.py) to check if GPUs are working correctly.


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


## Parallel processing
It is customary to run a function multiple times on different inputs (e.g., compute spectrograms from multiple audio
recordings, apply some processing to all the images in a dataset, analyze each frame of a video, etc.).
This can be done serially with a for loop, or in parallel over multiple cores.
Have a look at the example script [*run_parallel.py*](run_parallel.py).


## Additional resources
General
- [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Pandas](https://pandas.pydata.org/docs/getting_started/index.html)

Machine learning
- [scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)
- [Tensorflow / Keras](https://www.tensorflow.org/tutorials)
- [Pytorch](https://pytorch.org/tutorials/)
- [Tensorboard](https://www.tensorflow.org/tensorboard/get_started)

Image and video processing
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [scikit-image](https://scikit-image.org/)
- [OpenCV](https://opencv.org/)
- [imgaug](https://imgaug.readthedocs.io/en/latest/)
- [Albumentation](https://albumentations.ai/)
- [scikit-video](http://www.scikit-video.org/)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- CNN Segmentation Models both in [Tensorflow](https://github.com/qubvel/segmentation_models) and [PyTorch](https://github.com/qubvel/segmentation_models.pytorch)

Audio processing
- [librosa](https://librosa.org/)
- [Audiomentation](https://github.com/iver56/audiomentations)
- [pyroomacoustics](https://pyroomacoustics.readthedocs.io/)


## Credits
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Paolo Bestagini
- Francesco Picetti
- Nicolò Bonettini
- Francesco Maffezzoli
