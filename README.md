# Transformer-GCN-QA

A Q/A architecture based on transformers and GCNs.

## Installation

It is highly recommended that you first create a virtual environment. For example, using `conda`

```
$ conda create -n transformer-gcn-qa python=3 -y
$ conda activate transformer-gcn-qa
```

Next, follow the instructions [here](https://spacy.io/usage) to install spaCy and download the english language model. For example, using `conda`

```
$ conda install -c conda-forge spacy
$ python -m spacy download en_core_web_md
```

Finally, install all the other dependencies

```
$ pip install -r requirements.txt
```

## Usage

The main classes are outlined below

###  `Preprocessor`

This class provides methods for processing raw text. Most importantly, `Preprocessor.transform()` can be used to transform the Wiki- or MedHop datasets into a format that can then be used to construct a graph for learning with a GCN.

```python
from utils import load_wikihop
from preprocessor import Preprocessor

preprocessor = Preprocessor()
dataset = load_wikihop('../path/to/dataset')
model = None  # TODO: This will be a PyTorch implementation of BERT, for now, leave as None.

processed_dataset = preprocessor.transform(dataset, model)
```

`processed_dataset` is simply the loaded Wiki- or MedHop dataset with an added key for each training example, `'processed_supports'`, which contain a list (containing tokenized sentences) of lists (containing tokens) of 3-tuples: `(text, label, embedding)` where:

- `text`: is a given token in a given supporting document
- `label`: is an NER label assigned by `model`.
- `embedding`: is a contextual token embedding, assigned by `model`.

Note that if `model` is `None`, `label` and `embedding` are provided by a SpaCy language model.

## Troubleshooting

If you get an error mentioning `spacy.strings.StringStore size changed, may indicate binary incompatibility` you will need to install `neuralcoref` from the distribution's sources

```
$ pip uninstall neuralcoref
$ pip install neuralcoref --no-binary neuralcoref
```
