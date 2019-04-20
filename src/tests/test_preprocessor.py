class TestPreprocessor():
    """Collects all unit tests for `src.preprocessor.Preprocessor` a class for preprocessing Wiki-
    or MedHop.
    """
    def test_find_candidates_simple(self, preprocessor):
        """Given a simple example, asserts that `preprocessor._find_candidates` returns the expected
        values.
        """
        candidates = ['this meaningless string']
        supporting_doc = 'I want you to find this meaningless string.'

        expected = [(19, 42)]
        actual = preprocessor._find_candidates(candidates=candidates, supporting_doc=supporting_doc)

        assert expected == actual
        # Check that the returned indices return expected substring in supporting_doc
        assert supporting_doc[actual[0][0]:actual[0][-1]] == candidates[0]

    def test_process_doc_simple(self, preprocessor, model, embeddings):
        """Given a simple example, asserts that `preprocessor._process_doc` returns the expected
        values.
        """
        test = preprocessor.nlp('My sister has a dog. She loves him.')

        expected_tokens = ['My', 'sister', 'has', 'a', 'dog', '.', 'She', 'loves', 'him', '.']
        expected_offsets = [(0, 2), (3, 9), (10, 13), (14, 15), (16, 19), (19, 20), (21, 24), 
                            (25, 30), (31, 34), (34, 35)]
        expected_corefs = [[(0, 9), (21, 24)], [(14, 19), (31, 34)]]
        expected_embedding_indices = list(range(len(expected_tokens)))

        actual = preprocessor._process_doc(doc=test, model=model, embeddings=embeddings)

        (actual_tokens, actual_offsets, actual_corefs,
         actual_embeddings, actual_embedding_indices) = actual

        assert expected_tokens == actual_tokens
        assert expected_offsets == actual_offsets
        assert expected_corefs == actual_corefs
        assert expected_embedding_indices == actual_embedding_indices
        assert actual_embeddings.shape[0] == len(expected_tokens)
