import pytest
import spacy
import torch

from ..models import BERT
from ..preprocessor import Preprocessor


@pytest.fixture
def nlp():
    """Returns a loaded SpaCy model.
    """
    nlp = spacy.load('en_core_web_sm')
    
    return nlp

@pytest.fixture
def preprocessor(nlp):
    """Returns an instance of a Preprocessor object."""
    # Load the small spaCy model, which will be quicker for testing
    preprocessor = Preprocessor(nlp=nlp)

    return preprocessor

@pytest.fixture
def model():
    """Returns an instance of a BERT object."""
    bert = BERT()
    
    return bert

@pytest.fixture
def embeddings():
    """Returns an empty torch Tensor"""
    embeddings = torch.tensor([])
    
    return embeddings
