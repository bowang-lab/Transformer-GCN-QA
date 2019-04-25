import torch

# TODO: Write unit test that checks the vectors in actual_encoded_mentions are the same as
# those returned by BERT.


class TestPreprocessor():
    """Collects all unit tests for `src.preprocessor.Preprocessor` a class for preprocessing Wiki-
    or MedHop.
    """
    def test_transform_simple(self, dataset, preprocessor, model):
        """
        """
        (actual_processed_candidates, actual_encoded_mentions,
         actual_encoded_mentions_split_sizes, actual_candidate_idxs, actual_targets) = \
            preprocessor.transform(dataset, model)

        # TODO 1 Example should include corefs
        expected_processed_candidates = {
            'train': {
                'WH_train_0': [[]],
                'WH_train_1': [
                    [
                        {'mention': 'english', 'corefs': []},
                        {'mention': 'spanish', 'corefs': []},
                    ],
                    [
                        {'mention': 'nahuatl', 'corefs': []},
                        {'mention': 'spanish', 'corefs': []},
                    ]
                ]
            }
        }
        expected_encoded_mentions_split_sizes = torch.tensor([0, 4])
        expected_candidate_idxs = {
            'train': [
                {},
                {'english': [0],
                 'spanish': [1, 3],
                 'nahuatl': [2]}
            ]
        }
        expected_targets = {'train': [[1, 0, 0], [1, 0, 0, 0, 0]]}

        assert expected_processed_candidates == actual_processed_candidates
        # 4 because there are four mentions and 768 b/c it is the size of BERT encodings
        assert actual_encoded_mentions['train'].shape == (4, 768)
        assert torch.equal(expected_encoded_mentions_split_sizes,
                           actual_encoded_mentions_split_sizes['train'])
        assert expected_candidate_idxs == actual_candidate_idxs
        assert expected_targets == actual_targets

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
                'mention': 'my sister',
                'corefs': [
                    {
                        'mention': 'she',
                    }
                ]
            },
            {
                'mention': 'a dog',
                'corefs': [
                    {
                        'mention': 'him',
                    }
                ]
            },
        ]

        expected_candidate_idxs = {
            'my sister': [0, 1],
            'a dog': [2, 3]
        }

        tokens, offsets, corefs, embeddings = preprocessor._process_doc(doc, model)
        candidate_offsets = preprocessor._find_candidates(candidates, supporting_doc)

        (actual_processed_candidates, actual_candidate_idxs, actual_encoded_mentions,
         encoded_mention_idx) = \
            preprocessor._process_candidates(candidate_offsets=candidate_offsets,
                                             supporting_doc=supporting_doc,
                                             tokens=tokens,
                                             offsets=offsets,
                                             corefs=corefs,
                                             embeddings=embeddings,
                                             candidate_idxs={},)

        assert expected_processed_candidates == actual_processed_candidates
        assert expected_candidate_idxs == actual_candidate_idxs
        # 4 because there are two mentions, each with a coreferent mention, and 768 b/c it is the
        # size of BERT encodings
        assert len(actual_encoded_mentions) == 4
        assert torch.cat(actual_encoded_mentions, dim=0).shape == (4, 768)
        assert encoded_mention_idx == 4

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
