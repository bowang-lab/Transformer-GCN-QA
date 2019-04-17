# Transformer-GCN-QA

A Q/A architecture based on transformers and GCNs.

## Installation

It is highly recommended that you first create a virtual environment. For example, using `conda`

```
$ conda create -n transformer-gcn-qa python=3 -y
$ conda activate transformer-gcn-qa
```

Next, download the [spaCy](https://spacy.io/) english language model.

```
$ python -m spacy download en_core_web_md
```

Then follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch for your system. It is highly reccomended that you use a GPU to train the model (or preprocess Wiki- or MedHop) so make sure to install CUDA support. For example, using `conda` and installing for Linux or Windows with CUDA 10.0

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Finally, install all the other dependencies

```
$ pip install -r requirements.txt
```

## Usage

### Classes

The main classes are outlined below. Call `help()` on any method or class to see more usage information, for example

```python
>> from src.models import BERT

>> help(BERT)
>> Help on class BERT in module src.models:

class BERT(builtins.object)
 |  BERT(pretrained_model='bert-base-uncased')
 |  
 |  A pre-trained BERT model which can be used to assign embeddings to tokenized text.
 |
 |  Args:
 |      pretrained_model (str): Name of pretrained BERT model to load. Must be in
 |      `PRETRAINED_MODELS`.
 |  
 |  Raises:
 |      ValueError if `pretrained_model` not in `PRETRAINED_MODELS`.
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

processed_dataset = preprocessor.transform(dataset, model)
```

`processed_dataset` is a dictionary of dictionaries, keyed by partition and training example IDs from the loaded Wiki- or MedHop dataset. For each training example there is a list of dictionaries (one per supporting document) containing

- `'mention'`: the exact text of a candidate found in the supporting document
- `'embeddings'`: contextual token embeddings for the mention, assigned by `model`
- `'corefs'`: a list of coreferring mentions, which themselves are dictionaries with `'mention'` and `'embeddings'` keys.

### Command line interface (CLI)

Command line interfaces are provided for convience. Pass `--help` to any script to get more usage information, for example

```
>> python -m src.cli.preprocess_wikihop --help
>> usage: preprocess_wikihop.py [-h] [-i INPUT] [-o OUTPUT]

Creates a dictionary for the given Wiki- or MedHop dataset which contains
everything we need for graph construction. Saves the resulting dataset to
disk.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the Wiki- or MedHop dataset.
  -o OUTPUT, --output OUTPUT
                        Path to save the processed output for the Wiki- or
                        MedHop dataset.
```

#### `preprocess_wikihop.py`

This script will take the Wiki- or MedHop dataset and save a pickle to disk containing everything we need to assemble the graph

```
python -m src.cli.preprocess_wikihop -i path/to/dataset -o path/to/output
```

## Troubleshooting

If you have any question about our codes or methods, please open an issue.

### Installation error from NeuralCoref

If you get an error mentioning `spacy.strings.StringStore size changed, may indicate binary incompatibility` you will need to install `neuralcoref` from the distribution's sources

```
$ pip uninstall neuralcoref
$ pip install neuralcoref --no-binary neuralcoref
```
