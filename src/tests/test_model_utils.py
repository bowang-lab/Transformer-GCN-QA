from ..utils import model_utils


class TestModelUtils(object):
    """Collects all unit tests for `utils.model_utils`.
    """
    def test_preprocess_query_empty(self):
        """Asserts that `model_utils.preprocess_query` returns the expected value for an empty
        input.
        """
        query = ""

        expected = "?"
        actual = model_utils.preprocess_query(query)

        assert expected == actual

    def test_preprocess_query_wikihop(self):
        """Asserts that `model_utils.preprocess_query` returns the expected value for a sample query
        from WikiHop.
        """
        query = "located_in_the_administrative_territorial_entity ralph bunche park"

        expected = "ralph bunche park located in the administrative territorial entity?"
        actual = model_utils.preprocess_query(query)

        assert expected == actual

    def test_preprocess_query_medhop(self):
        """Asserts that `model_utils.preprocess_query` returns the expected value for a sample query
        from MedHop.
        """
        query = "interacts_with DB00563?"

        expected = "DB00563 interacts with?"
        actual = model_utils.preprocess_query(query)

        assert expected == actual
