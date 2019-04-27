import spacy
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from torch import nn
from torch_geometric.nn import RGCNConv

from .constants import CLS
from .constants import PAD
from .constants import PRETRAINED_BERT_MODELS
from .constants import SEP
from .constants import SPACY_MODEL
from .utils import model_utils


class BERT(object):
    """A pre-trained BERT model which can be used to assign embeddings to tokenized text.

    Attributes:
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
            only_cls (bool): If True, a tensor of shape (len(tokens) x 768) is returned,
                corresponding to the output of the final layer of BERT on the special sentence
                classification token (`CLS`) for each sentence in `tokens`. This can be thought of
                as a summary of the input sentence. Otherwise, a tensor of the same shape as
                `tokens` is returned. Defaults to False.

        Returns:
            A Tensor, containing the hidden states of the last layer in `self.model`, which
            corresponds to a contextualized token embedding for each token in `text` if `only_cls`
            is False, otherwise a tensor of shape (len(tokens) x 768) corresponding to the hidden
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

    Attributes:
        nlp (spacy.lang): Optional, SpaCy language model. If None, loads `constants.SPACY_MODEL`.
            Defaults to None.
    """
    def __init__(self, batch_size, n_rgcn_layers=7, rgcn_size=768, n_rgcn_bases=10, nlp=None):
        super().__init__()

        # an object for processing natural language
        self.nlp = nlp if nlp else spacy.load(SPACY_MODEL)

        # BERT instance associated with this model
        self.bert = BERT()

        # hyperparameters of the model
        self.batch_size = batch_size
        self.n_rgcn_layers = n_rgcn_layers
        self.rgcn_size = rgcn_size
        self.n_rgcn_bases = n_rgcn_bases  # TODO: figure out a good number for this, 10 is a guess

        # layers of the model
        self.fc = nn.Linear(1536, self.rgcn_size)

        # Instantiate R-GCN layers
        self.rgcn_layers = []
        for _ in range(self.n_rgcn_layers):
            self.rgcn_layers.append(RGCNConv(self.rgcn_size, self.rgcn_size, 4, self.n_rgcn_bases))

        # Add R-GCN layers to model
        for i, layer in enumerate(self.rgcn_layers):
            self.add_module('RGCN_{}'.format(i), layer)

        # Final affine transform
        self.fc_logits = nn.Linear(self.rgcn_size + 768, 1)

    def encode_query(self, query):
        """Encodes a query (`query`) using BERT (`self.bert`).

        Reorganizes the query into a subject-verb statment and embeds it using the
        loaded and pre-trained BERT model (`self.bert`). The embedding assigned to the [CLS] token
        (`constants.CLS`) by BERT is taken as the encoded query.

        Args:
            query (str): A string representing the input query from Wiki- or MedHop.

        Returns:
            A tensor of size 768 containing the encoded query.
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

    def encode_query_aware_mentions(self, encoded_query, encoded_mentions):
        """Returns the query aware mention encodings.

        Concatenates the encoded query (`encoded_query`) and encoded mentions (`encoded_mentions`)
        and pushes the resulting tensor through a dense layer (`self.fc_1`) to return the
        query aware mention encodings.

        Args:
            encoded_query (torch.Tensor): A tensor of shape 768, containing the encoded query.
            encoded_mentions (torch.Tensor): A tensor of shape (n x 768), containing the encoded
                mentions.

        Returns:
            The query aware mention encodings, derived by concatenating the encoded query with
            each encoded mention and pushing the resulting tensor through a dense layer
            (`self.fc_1`).
        """
        num_mentions = encoded_mentions.shape[0]

        concat_encodings = \
            torch.cat((encoded_query.expand(num_mentions, -1), encoded_mentions), dim=-1)

        # Push concatenated query and mention encodings through fc layer followed by leaky ReLU
        # activation to get our query_aware_mention_encoding
        query_aware_mention_encoding = nn.LeakyReLU(self.fc_1(concat_encodings))

        return query_aware_mention_encoding

    def forward(self, query, encoded_mention, graph, cand_idxs, target=None):
        """Hook for the forward pass of the model.

        Args:
            query (str): A string representing the input query from Wiki- or MedHop.
            encoded_mentions (torch.Tensor): A tensor of encoded mentions, of shape
                (num of encoded mentions x 768).
            graph (torch.Tensor): A 3xN tensor of graph edges in coordinate format
                (rows 1 and 2) and edge types (row 3).
            cand_idxs (dict): A dictionary containing candidate as key and list 
                of candidate mention indices as values. These indices identify which
                rows of `encoded_mentions` correspond to mention embeddings of the
                given candidate.
            target (torch.Tensor): TODO: is it a tensor or a list? One-hot encoding of
                the candidate corresponding to the correct answer.

        Returns:
            TODO.
        """
        encoded_query = self.encode_query(query)

        query_aware_mentions = self.encode_query_aware_mentions(encoded_query, encoded_mention)

        x = self.fc(query_aware_mentions)

        # Separate `graph` into edge tensor and edge relation type tensor
        edge_index = graph[[0, 1], :]
        edge_type = graph[2, :]

        rgcn_layer_outputs = []  # Holds the output feature tensor from each R-GCN layer
        for layer in self.rgcn_layers:
            x = layer(x, edge_index, edge_type)
            # Can add NL activations here if we want
            rgcn_layer_outputs.append(x)

        # Sum outputs from each R-GCN layer
        x = torch.sum(torch.stack(rgcn_layer_outputs), dim=0)  # N x self.rgcn_size

        # Concatenate summed R-GCN output with query
        x_query_cat = torch.cat([x, encoded_query.expand((len(x), -1))], dim=-1)

        logits = self.fc_logits(x_query_cat)  # N x 1

        # Compute the masked softmax based on available candidates
        masked_softmax = torch.zeros(len(cand_idxs))
        for i, idxs in enumerate(cand_idxs.values()):
            logits_masked_max = torch.max(logits[idxs])
            masked_softmax[i] = torch.exp(logits_masked_max)
        masked_softmax /= torch.sum(masked_softmax)

        # If target is provided compute loss, otherwise return `masked_softmax`
        if target is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(masked_softmax.view(-1, len(cand_idxs)), target.view(-1))
            return loss
        else:
            return masked_softmax
