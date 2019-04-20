import spacy
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch import nn

from .constants import PRETRAINED_BERT_MODELS, SPACY_MODEL, PAD, CLS, SEP

from .utils import model_utils


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

        # Load pre-trained model & tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model)

        # Place the model on a CUDA device if avaliable
        self.device, self.n_gpus = model_utils.get_device(self.model)

    def predict_on_tokens(self, tokens, only_cls=False):
        """Uses `self.tokenizer` and `self.model` to run a prediction step on `tokens`.

        Using the pre-trained BERT tokenizer (`self.tokenizer`) and the pre-trained BERT model
        (`self.model), runs a prediction step on a list of lists representing tokenized sentences
        (`tokens`). Returns the a tensor containing the hidden state of the last layer in
        `self.model`, which corresponds to a contextualized token embedding for each token in
        `text`.

        Args:
            tokens (list): List of lists containing tokenized sentences.
            only_cls (bool): If True, a tensor of size (len(tokens) x 768) is returned,
                corresponding to the output of the final layer of BERT on the special sentence
                classification token (`CLS`) for each sentence in `tokens`. This can be thought of
                as a summary of the input sentence. Otherwise, a tensor of the same shape as
                `tokens` is returned. Defaults to False.

        Returns:
            A Tensor, containing the hidden states of the last layer in `self.model`, which
            corresponds to a contextualized token embedding for each token in `text` if `only_cls`
            is False, otherwise a tensor of size (len(tokens) x 768) corresponding to the hidden
            state of the last layer in `self.model` for the special sentence classification token
            (`CLS`).
        """
        indexed_tokens, attention_masks, orig_to_bert_tok_map = self.process_tokenized_input(tokens)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_output, _ = self.model(indexed_tokens,
                                           token_type_ids=torch.zeros_like(indexed_tokens),
                                           attention_mask=attention_masks,
                                           output_all_encoded_layers=False)

        # If only_cls, return only the embedding from the special classfication token
        if only_cls:
            encoded_output = encoded_output[:, 0, :]

        return encoded_output, orig_to_bert_tok_map

    def process_tokenized_input(self, tokens):
        """Processes tokenized sentences (`tokens`) for inference with BERT.

        Given `tokens`, a list of list representing tokenized sentences, processes the tokens
        and returns a three-tuple of token indices, attention masks and an original token to BERT
        token map. The token indices and attention masks can be used for inference with BERT. The
        original token to BERT token map is a determinisitc mapping for each token in `tokens`
        to the returned, token indices (note this is required as the tokenization process creates
        sub-tokens).

        Returns:
            A three-tuple of token indices, attention masks, and a original token to BERT token map.
        """
        # Tokenize input
        bert_tokens, orig_to_bert_tok_map = self._wordpiece_tokenization(tokens)

        # Pad the sequences
        max_sent = len(max(bert_tokens, key=len))
        bert_tokens = [sent + [PAD] * (max_sent - len(sent)) for sent in bert_tokens]

        # Generate attention masks for pad values
        attention_masks = [[int(token == PAD) for token in sent] for sent in bert_tokens]

        # Convert token to vocabulary indices
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(sent) for sent in bert_tokens]

        # Convert inputs to PyTorch tensors, put them on same device as model
        indexed_tokens = torch.tensor(indexed_tokens).to(self.device)
        attention_masks = torch.tensor(attention_masks).to(self.device)

        # Sanity check
        assert indexed_tokens.shape == attention_masks.shape

        return indexed_tokens, attention_masks, orig_to_bert_tok_map

    def _wordpiece_tokenization(self, orig_tokens):
        """WordPiece tokenize pre-tokenized text (`orig_tokens`) for inference with BERT.

        Given a list of lists representing pre-tokenized sentences, tokenizes each token with the
        WordPiece tokenizer associated with this BERT model
        (`self.self.tokenizer.wordpiece_tokenizer`) and appends the classication token
        and sentence seperater token, `CLS` and `SEP`, to the beginning and end of the sequence
        respectively. Additionaly, maintains a determinisitc mapping for each token in `orig_tokens`
        to the returned, WordPiece tokenized tokens.

        Returns:
            A two-tuple containing WordPiece tokenized tokens from `orig_tokens` and a deterministic
            mapping for each index in `orig_tokens` to an index in the returned WordPiece tokens.

        Resources:
            - https://github.com/google-research/bert/blob/master/README.md#tokenization
        """
        bert_tokens = []
        orig_to_bert_tok_map = []

        for sent in orig_tokens:
            bert_tokens.append([CLS])
            orig_to_bert_tok_map.append([])
            for orig_token in sent:
                orig_to_bert_tok_map[-1].append(len(bert_tokens[-1]))
                bert_tokens[-1].extend(self.tokenizer.wordpiece_tokenizer.tokenize(orig_token))
            bert_tokens[-1].append(SEP)

        return bert_tokens, orig_to_bert_tok_map


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

        # layers of the model
        self.mention_encoder = torch.nn.LSTM(input_size=768,
                                             hidden_size=384,
                                             num_layers=1,
                                             dropout=0.3,
                                             bidirectional=True)

        self.fc_1 = nn.Linear(1536, 786)
        self.fc_2 = nn.Linear(786, 512)

    def encode_query(self, query):
        """Encodes a query (`q`) using BERT (`self.bert`).
        Reorganizes the query into a subject-verb statment and embeds it using the
        loaded and pre-trained BERT model (`self.bert`). The embedding assigned to the [CLS] token
        by BERT is taken as the encoded query.
        Args:
            q (str): A string representing the input query from Wiki- or MedHop.
        Returns:
            A vector of size 768 containing the encoded query.
        """
        # Reorganize query into a subject-verb arrangement
        query_sv = ' '.join(query.split(' ')[1:] + query.split(' ')[0].split('_'))

        # Get everything we need for inference with BERT
        indexed_tokens, attention_masks, _ = \
            self.bert.process_tokenized_input([[token.text for token in self.nlp(query_sv)]])

        # Push the query through BERT
        query_encoding, _ = self.bert.model(indexed_tokens,
                                            token_type_ids=torch.zeros_like(indexed_tokens),
                                            attention_mask=attention_masks,
                                            output_all_encoded_layers=False)

        # Use the embedding assigned to the [CLS] token as the encoded query
        query_encoding = query_encoding.squeeze(0)[0, :]

        return query_encoding

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

        # Extract the final forward/backward hidden states
        final_forward_hidden_state, final_backward_hidden_state = \
            self._get_forward_backward_hidden_states(hn, self.mention_encoder, batch_size=x.shape[1])

        # Concat final forward/backward hidden states yields (1, 2 * hidden_size) encoded mentions
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

    def forward(self, query, x):
        """TODO.
        """
        encoded_query = self.encode_query(query)
        encoded_mentions = self.encode_mention(x)

        query_aware_mentions = self.encode_query_aware_mentions(encoded_query, encoded_mentions)

        # TODO: Graph building, RGCNs, etc.

    def _get_forward_backward_hidden_states(self, hn, lstm, batch_size):
        """Helper function that returns the final forward/backward hidden states from `hn` given the
        LSTM that produced them (`lstm`).
        """
        # Reshape it in order to be able to extract final forward/backward hidden states
        num_layers = lstm.num_layers
        num_directions = 2 if lstm.bidirectional else 1
        hidden_size = lstm.hidden_size

        hn = hn.view(num_layers, num_directions, batch_size, hidden_size)

        final_forward_hidden_state = hn[-1, 0, :, :]
        final_backward_hidden_state = hn[-1, -1, :, :]

        return final_forward_hidden_state, final_backward_hidden_state
