import pytest
import spacy
from pkg_resources import resource_filename

from ..constants import SPACY_MODEL
from ..models import BERT
from ..models import TransformerGCNQA
from ..preprocessor import Preprocessor
from ..utils.dataset_utils import load_wikihop

# Constants
PATH_TO_DUMMY_DATASET = resource_filename(__name__, 'resources/dummy_dataset')

# Fixtures
@pytest.fixture
def dataset():
    """Returns a loaded dataset, which is structured like Wiki- or MedHop, with `masked==False` in
    call to `load_wikihop()`.
    """
    dataset = load_wikihop(PATH_TO_DUMMY_DATASET)

    return dataset


@pytest.fixture
def masked_dataset():
    """Returns a loaded dataset, which is structured like Wiki- or MedHop, where `masked==True` in
    call to `load_wikihop()`.
    """
    dataset = load_wikihop(PATH_TO_DUMMY_DATASET, masked=True)

    return dataset


@pytest.fixture
def nlp():
    """Returns a loaded SpaCy model.
    """
    nlp = spacy.load(SPACY_MODEL)

    return nlp


@pytest.fixture
def preprocessor(nlp):
    """Returns an instance of a Preprocessor object."""
    # Load the small spaCy model, which will be quicker for testing
    preprocessor = Preprocessor(nlp=nlp)

    return preprocessor


@pytest.fixture
def bert():
    """Returns an instance of a BERT object."""
    bert = BERT()

    return bert


@pytest.fixture
def transformer_gcn_qa(nlp):
    transformer_gcn_qa = TransformerGCNQA(nlp=nlp)

    return transformer_gcn_qa
