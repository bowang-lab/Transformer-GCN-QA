import torch

# TODO: Write unit test that checks the vectors in actual_encoded_mentions are the same as
# those returned by BERT.


class TestPreprocessor():
    """Collects all unit tests for `src.preprocessor.Preprocessor` a class for preprocessing Wiki-
    or MedHop.
    """
    def test_process_doc_simple(self, preprocessor, model):
        """Given a simple example, asserts that `preprocessor._process_doc` returns the expected
        values.
        """
        doc = preprocessor.nlp('My sister has a dog. She loves him.')

        expected_tokens = ['My', 'sister', 'has', 'a', 'dog', '.', 'She', 'loves', 'him', '.']
        expected_offsets = [(0, 2), (3, 9), (10, 13), (14, 15), (16, 19), (19, 20), (21, 24),
                            (25, 30), (31, 34), (34, 35)]
        expected_corefs = [[(0, 9), (21, 24)], [(14, 19), (31, 34)]]

        actual = preprocessor._process_doc(doc=doc, model=model)
        actual_tokens, actual_offsets, actual_corefs, actual_embeddings = actual

        assert expected_tokens == actual_tokens
        assert expected_offsets == actual_offsets
        assert expected_corefs == actual_corefs
        assert actual_embeddings.shape[0] == len(expected_tokens)

    def test_process_candidates_simple(self, preprocessor, model):
        """Given a simple example, asserts that `preprocessor._process_candidates` returns the
        expected values.
        """
        candidates = ['My sister', 'a dog']
        supporting_doc = 'My sister has a dog. She loves him.'
        doc = preprocessor.nlp('My sister has a dog. She loves him.')

        expected_processed_candidates = [
            {
                'mention': 'My sister',
                'encoding_idx': 0,
                'corefs': [
                    {
                        'mention': 'She',
                        'encoding_idx': 1,
                    }
                ]
            },
            {
                'mention': 'a dog',
                'encoding_idx': 2,
                'corefs': [
                    {
                        'mention': 'him',
                        'encoding_idx': 3
                    }
                ]
            },
        ]

        tokens, offsets, corefs, embeddings = preprocessor._process_doc(doc, model)
        candidate_offsets = preprocessor._find_candidates(candidates, supporting_doc)

        actual_processed_candidates, actual_encoded_mentions = \
            preprocessor._process_candidates(candidate_offsets=candidate_offsets,
                                             supporting_doc=supporting_doc,
                                             tokens=tokens,
                                             offsets=offsets,
                                             corefs=corefs,
                                             embeddings=embeddings)

        assert expected_processed_candidates == actual_processed_candidates
        assert len(actual_encoded_mentions) == 4
        # 4 because there are two mentions, each with a coreferent mention, and 768 b/c it is the
        # size of BERT encodings
        assert torch.cat(actual_encoded_mentions, dim=0).shape == (4, 768)

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
