# Transformer-GCN-QA

[![Build Status](https://travis-ci.com/berc-uoft/Transformer-GCN-QA.svg?branch=master)](https://travis-ci.com/berc-uoft/Transformer-GCN-QA)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e25aeff5b35046e3831c4517efe0b813)](https://app.codacy.com/app/JohnGiorgi/Transformer-GCN-QA?utm_source=github.com&utm_medium=referral&utm_content=berc-uoft/Transformer-GCN-QA&utm_campaign=Badge_Grade_Dashboard)

A Q/A architecture based on transformers and GCNs.

## Installation

This package requires python 3.7+ and CUDA 10.0. There are several dependencies not in the `setup.py` that you will need to install before installing this package. 

First, although optional, it is highly recommended that you create a virtual environment. For example, using `conda`

```bash
$ conda create -n transformer-gcn-qa python=3.7 -y
$ conda activate transformer-gcn-qa
# Notice, the prompt has changed to indicate that the enviornment is active
(transformer-gcn-qa) $ 
```

You will then need to install CUDA 10.0 by following the [installation instructions](https://docs.nvidia.com/cuda/index.html#installation-guides) for your system. CUDA 10.0 can be downloaded from [here](https://developer.nvidia.com/cuda-10.0-download-archive).

With CUDA 10.0 installed, follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch for your system. For example, using `conda` and installing for Linux or Windows with CUDA 10.0

```bash
(transformer-gcn-qa) $ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

The R-GCN implementation is from [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric). Installation involves ensuring various system variables are set followed by `pip` installing a number of packages. Comprehensive installation instructions can be found [here](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html).

Finally, install this package and its remaining dependencies straight from GitHub

```bash
(transformer-gcn-qa) $ pip install git+https://github.com/berc-uoft/Transformer-GCN-QA.git
```

or install by cloning this repository

```
(transformer-gcn-qa) $ git clone https://github.com/berc-uoft/Transformer-GCN-QA.git
(transformer-gcn-qa) $ cd Transformer-GCN-QA
```

and then using either `pip`

```bashen_core_web_lg
(transformer-gcn-qa) $ pip install -e .
```

 or `setuptools`

```bash
(transformer-gcn-qa) $ python setup.py install
```

Regardless of installation method, you will need to additionally download a [SpaCy](https://spacy.io/) English language model

```bash
(transformer-gcn-qa) $ python -m spacy download en_core_web_lg
```

### Install with development requirements

To run the test suite (or use [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)), you will want to install with

```
(transformer-gcn-qa) $ pip install -e .[dev]
```

## Usage

### Quickstart

Download Wiki- or MedHop [here](https://qangaroo.cs.ucl.ac.uk/), and then preprocess it with

```
python -m src.cli.preprocess_wikihop -i path/to/wiki/or/medhop -o path/to/preprocessed/wiki/or/medhop
```

> Note, this can take up to 12 hours to complete.

Then build the graphs used during the online training step

```
python -m src.cli.build_graphs -i path/to/preprocessed/wiki/or/medhop
```

Finally, to train the model

```
python -m src.cli.train -i path/to/preprocessed/wiki/or/medhop
```

See below for more detailed usage instructions.

### Command line interface (CLI)

Command line interfaces are provided for convenience. Pass `--help` to any script to get more usage information, for example

```
(transformer-gcn-qa) $ python -m src.cli.preprocess_wikihop --help
```

#### `preprocess_wikihop.py`

This script will take the Wiki- or MedHop dataset and save to disk everything we need to assemble the graph

```
(transformer-gcn-qa) $ python -m src.cli.preprocess_wikihop -i path/to/wiki/or/medhop -o path/to/preprocessed/wiki/or/medhop
```

#### `build_graphs.py`

This script will take the pre-processed Wiki- or MedHop dataset and create/save the graph tensors to disk, which are used as inputs to the model.

```
(transformer-gcn-qa) $ python -m src.cli.build_graph -i path/to/preprocessed/wiki/or/medhop
```

#### `train.py`

This script will train a model on a pre-processed Wiki- or MedHop dataset.

```
(transformer-gcn-qa) $ python -m src.cli.train -i path/to/preprocessed/wiki/or/medhop
```

To monitor performance with TensorBoard, first, make sure you have [installed with dev dependencies](#install-with-development-requirements). During a training session, call `tensorboard --logdir=runs` and then access port `6006` in your browser.

## Troubleshooting

If you have any question about our code or methods, please open an issue.

### Test suite

The test suite can be found in `src/tests`. To run, first follow the instructions for [installing with development requirements](#install-with-development-requirements)). 

The test suite can then be run with the following command

```
(transformer-gcn-qa) $ cd Transformer-GCN-QA
(transformer-gcn-qa) $ tox
```

### Installation error from NeuralCoref

If you get an error mentioning `spacy.strings.StringStore size changed, may indicate binary incompatibility` you will need to install `neuralcoref` from the distribution's sources

```
(transformer-gcn-qa) $ pip uninstall neuralcoref -y
(transformer-gcn-qa) $ pip install neuralcoref --no-binary neuralcoref
```
