import pytest
import spacy
from pkg_resources import resource_filename

from ..models import BERT
from ..preprocessor import Preprocessor
from ..utils.dataset_utils import load_wikihop

# Constants
PATH_TO_DUMMY_DATASET = resource_filename(__name__, 'resources/dummy_dataset')

# Fixtures
@pytest.fixture
def dataset():
    """Returns a loaded dataset, which is structured like Wiki- or MedHop
    """
    dataset = load_wikihop(PATH_TO_DUMMY_DATASET)

    return dataset


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
