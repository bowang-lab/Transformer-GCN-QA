import pytest
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer

from ..constants import CLS
from ..constants import SEP
from ..models import BERT


class TestBert(object):
    """Collects all unit tests for `model.BERT`, a PyTorch implementation of BERT.
    """
    def test_init(self, bert):
        """Asserts that class attributes are as expected after initialization.
        """
        assert isinstance(bert.tokenizer, BertTokenizer)
        assert isinstance(bert.model, BertModel)
        assert bert.model.eval

        assert bert.device.type == 'cpu'

    def test_invalid_pretrained_model_value_error(self):
        """Asserts that `BERT` throws a ValueError when an invalid value for `pretrained_model` is
        passed."""
        with pytest.raises(ValueError):
            BERT(pretrained_model='this should throw a ValueError!')

    def test_predict_on_tokens_simple_full_sequence(self, bert):
        """Asserts that `BERT.predict_on_tokens` returns the expected value for a simple
        input when `only_cls=False`.
        """
        orig_tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]

        expected_orig_to_bert_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]

        actual_endcoded_output, actual_orig_to_bert_tok_map = \
            bert.predict_on_tokens(orig_tokens, only_cls=False)

        assert actual_endcoded_output.shape == (2, 8, 768)
        assert expected_orig_to_bert_tok_map == actual_orig_to_bert_tok_map

    def test_predict_on_tokens_simple_only_cls(self, bert):
        """Asserts that `BERT.predict_on_tokens` returns the expected value for a simple
        input when `only_cls=True`.
        """
        orig_tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]

        expected_orig_to_bert_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]

        actual_endcoded_output, actual_orig_to_bert_tok_map = \
            bert.predict_on_tokens(orig_tokens, only_cls=True)

        assert actual_endcoded_output.shape == (2, 768)
        assert expected_orig_to_bert_tok_map == actual_orig_to_bert_tok_map

    def test_process_tokenized_input_simple(self, bert):
        """Asserts that `BERT.process_tokenized_input` returns the expected value for a simple
        input.
        """
        orig_tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]

        expected_attention_masks = torch.tensor([
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 0.]
        ])
        expected_orig_to_bert_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]

        (actual_indexed_tokens, actual_orig_to_bert_tok_map, actual_attention_masks) = \
            bert.process_tokenized_input(orig_tokens)

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, 8)
        assert torch.equal(expected_attention_masks, actual_attention_masks)
        assert expected_orig_to_bert_tok_map == actual_orig_to_bert_tok_map

    def test_wordpiece_tokenization_simple(self, bert):
        """Asserts that `BERT._wordpiece_tokenization_simple` returns the expected value for a
        simple input.
        """
        orig_tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]

        expected_bert_tokens = [
            [CLS, "john", "johan", "##son", "'", "##s",  "house", SEP],
            [CLS, "who", "was", "jim", "henson", "?", SEP]
        ]
        expected_orig_to_bert_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]

        actual_bert_tokens, actual_orig_to_bert_tok_map = bert._wordpiece_tokenization(orig_tokens)

        assert expected_bert_tokens == actual_bert_tokens
        assert actual_orig_to_bert_tok_map == expected_orig_to_bert_tok_map


class TestTransformerGCNQA(object):
    """Collects all unit tests for `model.TransformerGCNQA`, a PyTorch QA model.
    """
    def test_init(self, nlp, transformer_gcn_qa):
        """Asserts that class attributes are as expected after initialization.
        """
        assert transformer_gcn_qa.nlp == nlp
        assert transformer_gcn_qa.dropout_rate == 0.2
        assert transformer_gcn_qa.n_rgcn_layers == 3
        assert transformer_gcn_qa.n_rels == 4
        assert transformer_gcn_qa.rgcn_size == 128
        assert transformer_gcn_qa.n_rgcn_bases == 2

        # TODO (John): Need tests for attributes not passed to init

    def test_encode_query_shape(self, transformer_gcn_qa):
        """Asserts that the shape of the tensor returned by `TransformerGCNQA.encode_query` is as
        expected.
        """
        query = "country_of_citizenship farid ahmadi"

        query_encoding = transformer_gcn_qa.encode_query(query)

        # Embedding will depend on specific BERT model used, so just check shape
        assert query_encoding.shape == (1, 768)
