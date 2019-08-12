import spacy
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from torch import nn

from torch_geometric.nn import RGCNConv

from .constants import CLS
from .constants import PAD
from .constants import PRETRAINED_BERT_MODEL
from .constants import SEP
from .constants import SPACY_MODEL
from .utils import model_utils
from .utils.model_utils import get_device


class BERT(object):
    """A pre-trained BERT model which can be used to assign embeddings to tokenized text.

    Attributes:
        pretrained_model (str): Name of pretrained BERT model to load. Must be in
        `PRETRAINED_BERT_MODELS`.

    Raises:
        ValueError if `pretrained_model` not in `PRETRAINED_BERT_MODELS`.
    """
    def __init__(self, pretrained_model=PRETRAINED_BERT_MODEL):
        # Load pre-trained model & tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
        self.model = BertModel.from_pretrained(pretrained_model)
        # This model will always be used for inference
        self.model.eval()

        # Place the model on a CUDA device if avaliable
        self.device, _, = model_utils.get_device(self.model)

    def predict_on_tokens(self, tokens):
        """Uses `self.tokenizer` and `self.model` to run a prediction step on `tokens`.

        Using the pre-trained BERT tokenizer (`self.tokenizer`) and the pre-trained BERT model
        (`self.model`), runs a prediction step on a list of lists representing tokenized sentences
        (`tokens`). Returns a three tuple of:

            - `pooled_output`: the tensor representing the sequence classification token (`CLS`)
            - `encoded_output`: the tensor containing the hidden state of the last layer in
              `self.model`, which corresponds to a contextualized token embedding for each token in
              `text`.
            - `orig_to_bert_tok_map`: a list containing a deterministic mapping for each index in
              `tokens` to an index in the WordPiece tokens that passed to BERT.

        Args:
            tokens (list): List of lists containing tokenized sentences.

        Returns:
            A three-tuple containing `pooled_output`, `encoded_output` and `orig_to_bert_tok_map`.
        """
        indexed_tokens, orig_to_bert_tok_map, attention_masks = \
            self.process_tokenized_input(tokens)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_output, pooled_output = \
                self.model(indexed_tokens,
                           token_type_ids=torch.zeros_like(indexed_tokens),
                           attention_mask=attention_masks,
                           output_all_encoded_layers=False)

        return pooled_output, encoded_output, orig_to_bert_tok_map

    def process_tokenized_input(self, tokens):
        """Processes tokenized sentences (`tokens`) for inference with BERT.

        Returns a three-tuple of token indices, attention masks and an
        original token to BERT token map. The token indices and attention masks can be used for
        inference with BERT. The original token to BERT token map is a determinisitc mapping for
        each token in `tokens` to the returned token indices (this is required as the tokenization
        process creates sub-tokens).

        Returns:
            A three-tuple of token indices, an original token to BERT token map, and
            attention masks.
        """
        # Tokenize input
        bert_tokens, orig_to_bert_tok_map = self._wordpiece_tokenization(tokens)

        # Pad the sequences
        max_sent = len(max(bert_tokens, key=len))
        bert_tokens = [sent + [PAD] * (max_sent - len(sent)) for sent in bert_tokens]

        # Convert token to vocabulary indices
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(sent) for sent in bert_tokens]
        indexed_tokens = torch.tensor(indexed_tokens).to(self.device)

        # Generate attention masks for pad values
        attention_masks = [[float(idx > 0) for idx in sent] for sent in indexed_tokens]
        attention_masks = torch.tensor(attention_masks).to(self.device)

        return indexed_tokens, orig_to_bert_tok_map, attention_masks

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
    def __init__(self,
                 nlp=None,
                 pretrained_bert_model=PRETRAINED_BERT_MODEL,
                 dropout_rate=0.2,
                 n_rgcn_layers=3,
                 n_rels=3,
                 rgcn_size=512,
                 n_rgcn_bases=2, **kwargs):
        super().__init__()
        # TODO (John): This function is called multiple times. Find way to call it once.
        self.device, _ = get_device(self)

        # An object for processing natural language
        self.nlp = nlp if nlp else spacy.load(SPACY_MODEL)

        # BERT instance associated with this model
        self.bert = BERT(pretrained_model=pretrained_bert_model)

        # Hyperparameters of the model
        self.dropout_rate = dropout_rate

        self.n_rgcn_layers = n_rgcn_layers
        self.n_rels = n_rels
        self.rgcn_size = rgcn_size
        self.n_rgcn_bases = n_rgcn_bases

        # Layers of the model
        self.fc_1 = nn.Linear(1536, 786)
        self.fc_2 = nn.Linear(786, self.rgcn_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.rgcn_layer = RGCNConv(self.rgcn_size, self.rgcn_size, self.n_rels, self.n_rgcn_bases)


        # self.fc_3 = nn.Linear(self.rgcn_size + 768, 256)
        # self.fc_4 = nn.Linear(256, 128)
        # self.fc_5 = nn.Linear(128, 1)

        ### initialize the decoder's weights as the query encoding
        self.decoder_fc = nn.Linear(768, self.rgcn_size)
        # something goes here
        self.fc_3 = nn.Linear(self.rgcn_size, 128)
        self.fc_4 = nn.Linear(128, 1)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def encode_query(self, query):
        """Encodes a query (`query`) using BERT (`self.bert`).

        Reorganizes the query into a subject-verb statment and embeds it using the
        loaded and pre-trained BERT model (`self.bert`). The embedding assigned to the [CLS] token
        by BERT is taken as the encoded query.

        Args:
            query (str): A string representing the input query from Wiki- or MedHop.

        Returns:
            A tensor of size 768 containing the encoded query.
        """
        # Preprocess query to look more like natural language and tokenize
        query_svo = model_utils.preprocess_query(query)
        tokenized_query_svo = [token.text for token in self.nlp(query_svo)]

        # Push the query through BERT
        query_encoding, _, _ = self.bert.predict_on_tokens([tokenized_query_svo])

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
        query_aware_mention_encoding = self.leaky_relu(self.fc_1(concat_encodings))
        query_aware_mention_encoding = self.dropout(query_aware_mention_encoding)
        query_aware_mention_encoding = self.leaky_relu(self.fc_2(query_aware_mention_encoding))
        query_aware_mention_encoding = self.dropout(query_aware_mention_encoding)

        return query_aware_mention_encoding

    def forward(self, query, candidate_indices, encoded_mention, graph, target=None):
        """Hook for the forward pass of the model.

        Args:
            query (str): A string representing the input query from Wiki- or MedHop.
            candidate_indices (dict): A dictionary containing candidate as key and list of candidate
                mention indices as values. These indices identify which rows of `encoded_mentions`
                correspond to mention embeddings of the given candidate.
            encoded_mentions (torch.Tensor): A tensor of encoded mentions, of shape (num of encoded
                mentions x 768).
            graph (torch.Tensor): A 3 x N tensor of graph edges in coordinate format (rows 1 and 2)
                and edge types (row 3).
            target (torch.Tensor): One-hot encoding of the candidate corresponding to the correct
                answer.

        Returns:
            If target is not None, returns a single scalar representing the cross-entropy loss.
            Othewrise, returns a masked softmax prediction over all candidates in
            `candidate_indices`.
        """
        encoded_query = self.encode_query(query)

        x = self.encode_query_aware_mentions(encoded_query, encoded_mention)

        # Separate graph into edge tensor and edge relation type tensor
        edge_index = graph[[0, 1], :]
        edge_type = graph[2, :]

        rgcn_layers_sum = torch.zeros_like(x)
        for _ in range(self.n_rgcn_layers):
            x = self.rgcn_layer(x, edge_index, edge_type)
            # TODO (Duncan): Can add NL activations here if we want
            rgcn_layers_sum += x

        x = rgcn_layers_sum

        # Concatenate summed R-GCN output with query
        # x_query_cat = torch.cat([x, encoded_query.expand((len(x), -1))], dim=-1)

        # Affine transformations
        logits = self.fc_3(x)
        # query bmm logits instead of cat
        logits = torch.bmm(logits, encoded_query) # hope the dims match up
        logits = self.dropout(logits)
        logits = self.fc_4(logits)
        # logits = self.dropout(logits)
        # logits = self.fc_5(logits)  # N x 1, where N = # of candidates

        # Compute the masked softmax based on available candidates
        masked_logits = torch.zeros(len(candidate_indices)).to(self.device)
        for i, idxs in enumerate(candidate_indices.values()):
            if idxs:
                masked_logits[i] = torch.max(logits[idxs])

        # If target is provided return loss as well as logits
        if target is not None:
            loss_fct = nn.CrossEntropyLoss()
            class_index = torch.argmax(target, 1)
            loss = loss_fct(masked_logits.view(-1, len(candidate_indices)), class_index)
            return masked_logits, loss
        else:
            return masked_logits
