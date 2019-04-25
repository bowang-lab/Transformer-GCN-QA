# Transformer-GCN-QA

[![Build Status](https://travis-ci.com/berc-uoft/Transformer-GCN-QA.svg?branch=master)](https://travis-ci.com/berc-uoft/Transformer-GCN-QA)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e25aeff5b35046e3831c4517efe0b813)](https://app.codacy.com/app/JohnGiorgi/Transformer-GCN-QA?utm_source=github.com&utm_medium=referral&utm_content=berc-uoft/Transformer-GCN-QA&utm_campaign=Badge_Grade_Dashboard)

A Q/A architecture based on transformers and GCNs.

## Installation

It is highly recommended that you first create a virtual environment. For example, using `conda`

```
$ conda create -n transformer-gcn-qa python=3 -y
$ conda activate transformer-gcn-qa
# Notice, the prompt has changed to indicate that the enviornment is active
(transformer-gcn-qa) $ 
```

Next, download the [SpaCy](https://spacy.io/) english language model.

```
(transformer-gcn-qa) $ python -m spacy download en_core_web_md
```

Then follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch for your system. It is highly recommended that you use a GPU to train the model (or preprocess Wiki- or MedHop) so make sure to install CUDA support. For example, using `conda` and installing for Linux or Windows with CUDA 10.0

```
(transformer-gcn-qa) $ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

The R-GCN implementation is from [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric). Setup involves ensuring various system variables are set followed by `pip` installing a number of packages. Comprehensive installation instructions can be found [here](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html).

Finally, install this package straight from GitHub

```
(transformer-gcn-qa) $ pip install git+https://github.com/berc-uoft/Transformer-GCN-QA.git
```

or install by cloning the repository

```
(transformer-gcn-qa) $ git clone https://github.com/berc-uoft/Transformer-GCN-QA.git
(transformer-gcn-qa) $ cd Transformer-GCN-QA
```

and then using either `pip`

```
(transformer-gcn-qa) $ pip install -e .
```

 or `setuptools`

```
(transformer-gcn-qa) $ python setup.py install
```

### Install with development requirements

To run the test suite, you will want to install with

```
(transformer-gcn-qa) $ pip install -e .[dev]
```

## Usage

### Classes

The main classes are outlined below. Call `help()` on any method or class to see more usage information, for example

```
>> from src.models import BERT
>> help(BERT)
```

#### `Preprocessor`

This class provides methods for processing raw text. Most importantly, `Preprocessor.transform()` can be used to transform the Wiki- or MedHop datasets into a format that can then be used to construct a graph for learning with a GCN

```python
from src.utils.datasets import load_wikihop
from src.preprocessor import Preprocessor
from src.models import BERT

preprocessor = Preprocessor()
dataset = load_wikihop('../path/to/dataset')
model = BERT()

processed_candidates, encoded_mentions, encoded_mentions_split_sizes, candidate_idxs, targets = preprocessor.transform(dataset, model)
```

The returned tuple contains 5 dictionaries, keyed by dataset partition, with everything we need for graph construction and training. See `help(Preprocessor.transform)` for more information about each object

#### `BuildGraph`

This class constructs a heuristic graph for each provided sample. `BuildGraph` is instantiated with the `processed_dataset` output from `Preprocessor` as described above. Calling `BuildGraph.build()` then constructs a graph for each sample in `processed_dataset`.

```python
from src.preprocessor_graph import BuildGraph

buildGraph = BuildGraph(processed_dataset)
graphs, idxs = buildGraph.build()
```

`graphs` is a `3xN` tensor, where `N` is the sum of the number of edges across all graphs in `processed_dataset`. That is, if there are `K` samples in `processed_dataset`, then `N = N_1 + N_2 + ... + N_K` where `N_i` is the number of edges in the graph for sample i. 
- The first two rows of `graphs` correspond to the i-th and j-th indices of an edge respectively.
- The third row corresponds to the relation type of the edge (an integer in `[0, 1, 2, 3]`). 
The edges and corresponding relation types for each graph are concatenated in the second dimension to ensure a single tensor can be saved for all samples.

`idxs` is a dictionary containing sample ids as keys and corresponding graph tensor sizes as values. This allows the correct subtensor corresponding to the given sample from `graphs` to be extracted during training or inference time by using `torch.split()`.

### Command line interface (CLI)

Command line interfaces are provided for convenience. Pass `--help` to any script to get more usage information, for example

```
(transformer-gcn-qa) $ python -m src.cli.preprocess_wikihop --help
```

#### `preprocess_wikihop.py`

This script will take the Wiki- or MedHop dataset and save a pickle to disk containing everything we need to assemble the graph

```
(transformer-gcn-qa) $ python -m src.cli.preprocess_wikihop -i path/to/dataset -o path/to/output
```

#### `build_graph.py`

This script will take the processed Wiki- or MedHop pickle and save the graph tensor and size indices to disk, which are then used as inputs to the model.

```
(transformer-gcn-qa) $ python -m src.cli.build_graph -i path/to/processed/dataset -o path/to/output/
```

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
(transformer-gcn-qa) $ pip uninstall neuralcoref
(transformer-gcn-qa) $ pip install neuralcoref --no-binary neuralcoref
```
