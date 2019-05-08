import torch

# TODO: Write unit test that checks the vectors in actual_encoded_mentions are the same as
# those returned by BERT.


class TestPreprocessor():
    """Collects all unit tests for `src.preprocessor.Preprocessor` a class for preprocessing Wiki-
    or MedHop.
    """
    def test_transform_simple(self, dataset, preprocessor, bert):
        """Given a simply example, asserts that `preprocess.transform()` returns the expected
        values.
        """
        (actual_processed_dataset, actual_encoded_mentions, actual_encoded_mentions_split_sizes,
         actual_targets, actual_targets_split_sizes) = \
            preprocessor.transform(dataset, bert)

        # TODO 1 Example should include corefs
        expected_processed_dataset = {
            'train': {
                'WH_train_0': {
                    'mentions': [[]],
                    'query': "participant_of juan rossell",
                    'candidate_indices': {
                        '1996 summer olympics': [],
                        'olympic games': [],
                        'sport': [],
                    }
                },
                'WH_train_1': {
                    'mentions': [
                        [
                            {'text': 'english', 'corefs': []},
                            {'text': 'spanish', 'corefs': []},
                        ],
                        [
                            {'text': 'nahuatl', 'corefs': []},
                            {'text': 'spanish', 'corefs': []},
                        ]
                    ],
                    'query': "languages_spoken_or_written john osteen",
                    'candidate_indices': {
                        'english': [0],
                        'greek': [],
                        'koine greek': [],
                        'nahuatl': [2],
                        'spanish': [1, 3],
                    }
                }
            }
        }
        expected_encoded_mentions_split_sizes = {'train': [0, 4]}
        expected_targets = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0])
        expected_targets_split_sizes = {'train': [3, 5]}

        assert expected_processed_dataset == actual_processed_dataset
        # 4 because there are four mentions and 768 b/c it is the size of BERT encodings
        assert actual_encoded_mentions['train'].shape == (4, 768)
        assert expected_encoded_mentions_split_sizes == actual_encoded_mentions_split_sizes
        assert torch.equal(expected_targets, actual_targets['train'])
        assert expected_targets_split_sizes, actual_targets_split_sizes['train']

    def test_process_doc_simple(self, preprocessor, bert):
        """Given a simple example, asserts that `preprocessor._process_doc` returns the expected
        values.
        """
        doc = preprocessor.nlp('My sister has a dog. She loves him.')

        expected_tokens = ['my', 'sister', 'has', 'a', 'dog', '.', 'she', 'loves', 'him', '.']
        expected_offsets = [(0, 2), (3, 9), (10, 13), (14, 15), (16, 19), (19, 20), (21, 24),
                            (25, 30), (31, 34), (34, 35)]
        expected_corefs = [[(0, 9), (21, 24)], [(14, 19), (31, 34)]]

        actual = preprocessor._process_doc(doc=doc, model=bert)
        actual_tokens, actual_offsets, actual_corefs, actual_embeddings = actual

        assert expected_tokens == actual_tokens
        assert expected_offsets == actual_offsets
        assert expected_corefs == actual_corefs
        assert actual_embeddings.shape[0] == len(expected_tokens)

    def test_process_candidates_simple(self, preprocessor, bert):
        """Given a simple example, asserts that `preprocessor._process_candidates` returns the
        expected values.
        """
        candidates = ['my sister', 'a dog', 'arbitrary test']
        supporting_doc = 'My sister has a dog. She loves him.'
        doc = preprocessor.nlp('My sister has a dog. She loves him.')

        expected_processed_candidates = [
            {
                'text': 'my sister',
                'corefs': [
                    {
                        'text': 'she',
                    }
                ]
            },
            {
                'text': 'a dog',
                'corefs': [
                    {
                        'text': 'him',
                    }
                ]
            },
        ]

        expected_candidate_idxs = {
            'my sister': [0, 1],
            'a dog': [2, 3],
            # Check candidate makes it into this dict even if it does not appear in supporting docs
            'arbitrary test': [],
        }

        tokens, offsets, corefs, embeddings = preprocessor._process_doc(doc, bert)
        candidate_offsets = preprocessor._find_candidates(candidates, supporting_doc)
        candidate_indices = {candidate: [] for candidate in candidates}

        (actual_processed_candidates, actual_candidate_idxs, actual_encoded_mentions,
         encoded_mention_idx) = \
            preprocessor._process_candidates(candidate_offsets=candidate_offsets,
                                             supporting_doc=supporting_doc,
                                             tokens=tokens,
                                             offsets=offsets,
                                             corefs=corefs,
                                             embeddings=embeddings,
                                             candidate_indices=candidate_indices)

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
        # Check that we match at beginning, inside, and end of string and that matching is
        # case insensitive
        candidates = ['i want', 'this string', 'this other string.']
        supporting_doc = 'I want you to find this string and this other string.'

        expected = [(0, 6), (19, 30), (35, 53)]
        actual = preprocessor._find_candidates(candidates=candidates, supporting_doc=supporting_doc)

        assert expected == actual
        # Check that the returned indices return expected substring in supporting_doc
        assert supporting_doc[actual[0][0]:actual[0][-1]].lower() == candidates[0]
        assert supporting_doc[actual[1][0]:actual[1][-1]].lower() == candidates[1]
        assert supporting_doc[actual[2][0]:actual[2][-1]].lower() == candidates[2]
