from itertools import chain

import spacy
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch import nn

from constants import PRETRAINED_BERT_MODELS, SPACY_MODEL

from .utils import model_utils
from constants import PAD, CLS, SEP

# TODO: Figure out why torch.float16 is constantly mentioned as being faster (smaller ofc?).

class BERT():
    """A pre-trained BERT model which can be used to assign embeddings to tokenized text.

    Args:
        pretrained_model (str): Name of pretrained BERT model to load. Must be in
        `PRETRAINED_BERT_MODELS`.

    Raises:
        ValueError if `pretrained_model` not in `PRETRAINED_BERT_MODELS`.
    """
    def __init__(self, pretrained_model='bert-base-uncased'):
        if pretrained_model not in PRETRAINED_BERT_MODELS:
            err_msg = ("Expected `pretrained_model` to be one of {}."
                       " Got: {}".format(', '.join(PRETRAINED_BERT_MODELS), pretrained_model))
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
        num_sents = len(tokens)

        tokenized_text, indexed_tokens, attention_masks = self._process_tokenized_input(tokens)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_output, _ = self.model(indexed_tokens,
                                           token_type_ids=torch.zeros_like(indexed_tokens),
                                           attention_mask=attention_masks,
                                           output_all_encoded_layers=False)

        print('Encoded output: ', encoded_output.shape)
        wordpiece_mask = [[not token.startswith('##') for token in sent] for sent in tokenized_text]
        wordpiece_mask = torch.tensor(wordpiece_mask, dtype=torch.uint8).unsqueeze(-1).expand(-1, -1, 768)

        print('Wordpiece mask: ', wordpiece_mask.shape)

        # Features from last layer of BERT, excluding WordPiece tokens
        contextualized_embeddings = encoded_output.masked_select(wordpiece_mask)
        print('contextualized_embeddings: ', contextualized_embeddings.shape)
        contextualized_embeddings = contextualized_embeddings.view(num_sents, -1, 768)

        return contextualized_embeddings

    def process_tokenized_input(self, tokens):
        """Processes tokenized sentences (`tokens`) for inference with BERT.

        Given `tokens`, a list of list representing tokenized sentences, processes the tokens
        and returns a three-tuple of WordPiece tokenized text, token indices, and attention masks. 
        The token indices and attention masks can be used for inference with BERT.

        Returns:
            A three-tuple of WordPiece tokenized text, token indices, and attention masks.
        """
        # Tokenize input
        wp_tokenizer = self.tokenizer.wordpiece_tokenizer
        tokenized_text = [
            [[CLS]] + [wp_tokenizer.tokenize(token) for token in sent] + [[SEP]]
            for sent in tokens
        ]

        # Flatten to 2D list
        tokenized_text = [list(chain.from_iterable(sent)) for sent in tokenized_text]

        # Pad the sequences
        max_sent = len(max(tokenized_text, key=len))
        tokenized_text = [sent + [PAD] * (max_sent - len(sent)) for sent in tokenized_text]

        # Generate attention masks for pad values
        attention_masks = [[int(token == PAD) for token in sent] for sent in tokenized_text]

        # Convert token to vocabulary indices
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]

        # Convert inputs to PyTorch tensors
        indexed_tokens = torch.tensor(indexed_tokens).to(self.device)
        attention_masks = torch.tensor(attention_masks).to(self.device)

        # Sanity check
        assert indexed_tokens.shape == attention_masks.shape

        return tokenized_text, indexed_tokens, attention_masks


class TransformerGCNQA(nn.Module):
    """An end-to-end neural question answering achitecture based on transformers and GCNs.
    """
    def __init__(self, batch_size, nlp=None):
        super().__init__()

        # an object for processing natural language
        self.nlp = nlp if nlp else spacy.load(SPACY_MODEL)

        # BERT instance associated with this model
        self.bert = BERT()

        # hyperparameters of the model
        self.batch_size = batch_size

        self.fc_1 = nn.Linear(1536, 768)
        self.fc_2 = nn.Linear(768, 512)

    def encode_query(self, q):
        """Encodes a query (`q`) using BERT (`self.bert`).

        Reorganizes the query into a subject-verb statment and embeds it using the
        loaded and pre-trained BERT model (`self.bert`). The embedding assigned to the [CLS] token
        by BERT is taken as the encoded query.

        Args:
            q (str): A string representing the input query from Wiki- or MedHop.

        Returns:
            A vector of size (1, 768) containing the encoded query.
        """
        # Reorganize query into a subject-verb arrangement
        query_sv = ' '.join(q.split(' ')[1:] + q.split(' ')[0].split('_'))

        # Get everything we need for inference with BERT
        _, indexed_tokens, attention_masks = \
            self.bert.process_tokenized_input([[token.text for token in self.nlp(query_sv)]])

        # Push the query through BERT
        query_encoding = self.bert.model(indexed_tokens,
                                         token_type_ids=torch.zeros_like(indexed_tokens),
                                         attention_mask=attention_masks,
                                         output_all_encoded_layers=False).squeeze(0)

        # Use the embedding assigned to the [CLS] token as the encoded query
        return query_encoding[0, :]

    def encode_mentions(self, x):
        """Encodes a mention embedding (`x`) using the mention encoder (`self.mention_encoder`).

        Pushes mention embeddings (`x`) through a BiLSTM, (`self.mention_encoder`). The final hidden
        states of the forward and backward layers are concatenated to produce the encoded mentions.

        Args:
            x (torch.Tensor): Tensor of shape (seq_len, num_mentions, input_size) containing the
                BERT embeddings for mentions (`x`).

        Returns:
            A vector of size (num_mentions, 2 * `self.query_encoder.hidden_size`) containing the
            encoded mentions.
        """
        # hn is of shape (num_layers * directions, batch_size, hidden_size)
        _, (hn, _) = self.mention_encoder(x)

        # extract the final forward/backward hidden states
        final_forward_hidden_state, final_backward_hidden_state = \
            self._get_forward_backward_hidden_states(hn, self.mention_encoder, batch_size=x.shape[1])

        # concat final forward/backward hidden states yields (1, 2 * hidden_size) encoded mentions
        return torch.cat((final_forward_hidden_state, final_backward_hidden_state), dim=-1)

    def encode_query_aware_mentions(self, encoded_query, encoded_mentions):
        """Returns the query aware mention encodings.

        Concatenates the encoded query (`encoded_query`) and encoded mentions (`encoded_mentions`)
        and pushes the resulting tensor through the query mention encoder (`query_mention_enoder`),
        to return the query aware mention encodings.

        Args:
            TODO.

        Returns:
            TODO.
        """
        num_mentions = encoded_mentions.shape[0]
        concat_encodings = \
            torch.cat((encoded_query.expand(num_mentions, -1), encoded_mentions), dim=-1)

        return self.query_mention_encoder(concat_encodings)

    def forward(self, q, x):
        """TODO.
        """
        encoded_q = self.encode_query(q)
        encoded_mentions = self.encode_mention(x)

        query_aware_mentions = self.encode_query_aware_mentions(encoded_q, encoded_mentions)

        # TODO: Graph building, RGCNs, etc.

    def _get_forward_backward_hidden_states(self, hn, lstm, batch_size):
        """Helper function that returns the final forward/backward hidden states from `hn` given the
        LSTM that produced them (`lstm`).
        """
        # reshape it in order to be able to extract final forward/backward hidden states
        num_layers = lstm.num_layers
        num_directions = 2 if lstm.bidirectional else 1
        hidden_size = lstm.hidden_size

        hn = hn.view(num_layers, num_directions, batch_size, hidden_size)

        final_forward_hidden_state = hn[-1, 0, :, :]
        final_backward_hidden_state = hn[-1, -1, :, :]

        return final_forward_hidden_state, final_backward_hidden_state
