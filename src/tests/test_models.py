import pytest

from ..models import BERT


class TestBert(object):
    """Collects all unit tests for `model.BERT`, a PyTorch implementation of BERT.
    """
    def test_invalid_pretrained_model_value_error(self):
        """Asserts that `BERT` throws a ValueError when an invalid value for `pretrained_model` is
        passed."""
        with pytest.raises(ValueError):
            BERT(pretrained_model='this should throw a ValueError!')
