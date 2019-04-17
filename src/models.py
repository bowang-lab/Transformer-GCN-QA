from itertools import chain

import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

from .utils import model_utils

PRETRAINED_MODELS= [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
]

class BERT():
    """A pre-trained BERT model which can be used to assign embeddings to tokenized text.

    Args:
        pretrained_model (str): Name of pretrained BERT model to load. Must be in
        `PRETRAINED_MODELS`.

    Raises:
        ValueError if `pretrained_model` not in `PRETRAINED_MODELS`.
    """
    def __init__(self, pretrained_model='bert-base-uncased'):
        if pretrained_model not in PRETRAINED_MODELS:
            err_msg = ("Expected `pretrained_model` to be one of {}."
                       " Got: {}".format(', '.join(PRETRAINED_MODELS), pretrained_model))
            raise ValueError(err_msg)

        # load pre-trained model & tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model)

        # by default, place the model on the CPU
        self.device, self.n_gpus = model_utils.get_device(self.model)

    def predict_on_tokens(self, tokens):
        """Uses `self.tokenizer` and `self.model` to run a prediction step on `tokens`.

        Using the pre-trained BERT tokenizer (`self.tokenizer`) and the pre-trained BERT model,
        runs a prediction step on a list of tokens (`tokens`). Returns the a tensor containing
        the hidden state of the last layer in `self.model`, which corresponds to a contextualized
        token embedding for each token in `text`.

        Args:
            text (str): Raw text to be pushed through a pretrained BERT model.

        Returns:
            A Tensor, containing the hidden state of the last layer in `self.model`, which
            corresponds to a contextualized token embedding for each token in `text`.
        """
        # tokenize input, flatten to 1D list
        tokenized_text = [self.tokenizer.wordpiece_tokenizer.tokenize(token) for token in tokens]
        tokenized_text = list(chain.from_iterable(tokenized_text))

        # convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor)

        wordpiece_mask = torch.tensor(['##' not in token for token in tokenized_text])

        # features from last layer of BERT, excluding WordPiece tokens
        # drop the batch dim
        contextualized_embeddings = encoded_layers[-1][:, wordpiece_mask].squeeze(0)

        return contextualized_embeddings
