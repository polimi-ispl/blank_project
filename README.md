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
    ├── params.py                   # Simulation parameters
    └── README.md

## Getting started

### Prerequisites
- Install [conda](https://docs.conda.io/en/latest/miniconda.html)
- Create and activate the `py38_tf23` environment with *environment.yml*
```bash
$ conda env create -f environment.yml
$ conda activate py38_tf23
```

## Credits
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Paolo Bestagini