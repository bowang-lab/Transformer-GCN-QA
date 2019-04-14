import pytest

from ..preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    """Returns an instance of a Preprocessor object."""
    return Preprocessor()
