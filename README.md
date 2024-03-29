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
    ├── gpu_check_tf.py                 # Script to check the GPU setup on TF
    ├── gpu_check_torch.py              # Script to check the GPU setup on PyTorch
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
- Create and activate the `keras_env` environment with [*environment_keras.yml*](envs/environment_keras.yml)
```bash
conda env create -f envs/environment_keras.yml
conda activate keras_env
```

### Prepare a pytorch enviroment
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `torch_env` environment with [*environment_torch.yml*](envs/environment_torch.yml)
```bash
conda env create -f envs/environment_torch.yml
conda activate torch_env
```

### Use a GPU
Considering the high computational cost of deep learning algorithms, it is useful to accelarate on a GPU.
Basically, both tensorflow and pytorch come with native GPU support. However, they have different features:
- PyTorch natively supports 1 GPU at a time; if you (really) need multiple GPUs, you have to code your own
  `DataParallel` paradigm. To the best of our knowledge, it works perfectly at a glance.
- Tensorflow typically needs an extra effort, as it needs a [perfect combination](https://www.tensorflow.org/install/source#gpu) of driver/python/package versions.
  Run the script [*gpu_check_tf.py*](gpu_check_tf.py) to check if GPUs are working correctly.
  Note that, depending on the driver version, a server may or may not work with a given version of cuda.
  - Example: to use Tensorflow 2.3 (working on all machines), according to [this table](https://www.tensorflow.org/install/source#gpu) you should run in your environment
  ```bash
  conda install tensorflow==2.3.0 cudnn=7.6 cudatoolkit=10.1
  ```
  - Example: to use Tensorflow 2.6 (working on some machines), according to [this table](https://www.tensorflow.org/install/source#gpu) you should run in your environment
  ```bash
  conda install tensorflow==2.6.0 cudnn=8.1 cudatoolkit=11.2
  ```
  


## Train a CNN
To train a CNN
- Set the desired parameters in [*params.py*](params.py) (e.g., `model_name`, `input_shape`, etc.)
- Run the script [*train_cnn_keras.py*](train_cnn_keras.py) (if using Tensorflow / Keras) or
  [*train_cnn_pytorch.py*](train_cnn_pytorch.py) (if using pytorch) or
  [*train_cnn_lightning.py*](train_cnn_lightning.py) (if using pytorch lightning)
  
The trained model and training history are saved into `trained_models_root` folder defined in *params.py* (if using Tensorflow / Keras).

## Test a CNN
Once your network is trained, you want to test it on a new dataset.
Run the script [*test_cnn_pytorch.py*](test_cnn_pytorch.py) (if using pytorch) or
  [*test_cnn_lightning.py*](test_cnn_lightning.py) (if using pytorch lightning)

Check the use with `python test_cnn_pytorch.py --help`, you will see that you have to pass to the script the run folder that has been created by the training script.


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
This can be done serially with a for loop, or in parallel over multiple cores, or using multiple threads (best for I/O operations on lots of files).
Have a look at the example script [*run_parallel.py*](run_parallel.py).


## Additional resources
General
- [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html): useful conda commands.
- [PyCharm](https://www.jetbrains.com/pycharm/): python IDE ([free for students](https://www.jetbrains.com/community/education/#students)).
- [Pandas](https://pandas.pydata.org/docs/getting_started/index.html): data analysis and manipulation library.

Machine learning
- [scikit-learn](https://scikit-learn.org/stable/tutorial/index.html): machine learning library.
- [Tensorflow / Keras](https://www.tensorflow.org/tutorials): library for neural networks.
- [Pytorch](https://pytorch.org/tutorials/): library for neural networks.
- [Tensorboard](https://www.tensorflow.org/tensorboard/get_started): data log visualization tool.

Image and video processing
- [Pillow](https://pillow.readthedocs.io/en/stable/): image processing library.
- [scikit-image](https://scikit-image.org/): image processing library.
- [OpenCV](https://opencv.org/): computer vision library.
- [imgaug](https://imgaug.readthedocs.io/en/latest/): image augmentation library.
- [Augmentor](https://augmentor.readthedocs.io/en/master/): image augmentation library.
- [Albumentation](https://albumentations.ai/): image augmentation library.
- [scikit-video](http://www.scikit-video.org/): video processing library.
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python): ffmpeg python wrapper.
- CNN Segmentation Models both in [Tensorflow](https://github.com/qubvel/segmentation_models) and [PyTorch](https://github.com/qubvel/segmentation_models.pytorch).

Audio processing
- [librosa](https://librosa.org/): audio features library.
- [Audiomentation](https://github.com/iver56/audiomentations): audio augmentation library.
- [pyroomacoustics](https://pyroomacoustics.readthedocs.io/): library to generate room impulse responses.
- [Kapre](https://github.com/keunwoochoi/kapre): audio processing on GPU.
- [TorchAudio](https://pytorch.org/audio/stable/index.html): audio processing on GPU.


## Credits
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Paolo Bestagini
- Francesco Picetti
- Nicolò Bonettini
- Francesco Maffezzoli
