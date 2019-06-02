import re
from itertools import chain

import neuralcoref
import spacy
import torch
from tqdm import tqdm

from .constants import NEURALCOREF_GREEDYNESS
from .constants import SPACY_MODEL


class Preprocessor(object):
    """This class provides methods for processing the Wiki- and MedHop datasets. Most importantly,
    `Preprocessor.transform()` can be used to extract from the Wiki- or MedHop datasets everything
    we need for graph construction and training.

    Attributes:
        nlp (spacy.lang): Optional, SpaCy language model. If None, loads `constants.SPACY_MODEL`.
            Defaults to None.
        cased (bool): Optional, True if textual data should remain cased, otherwise lower cases it.
            Defaults to False.
    """
    def __init__(self, nlp=None):
        # SpaCy object for processing natural language
        self.nlp = nlp if nlp else spacy.load(SPACY_MODEL)
        # Adds coref to spaCy pipeline
        neuralcoref.add_to_pipe(self.nlp, greedyness=NEURALCOREF_GREEDYNESS)

    def transform(self, dataset, model=None):
        """Extracts `dataset` everything needed for graph construction and training.

        For the given Wiki- or MedHop `dataset`, returns a 5-tuple containing everything we need
        for graph construction and training. Each item in the tuple is a dictionary keyed by
        a dataset partition name (e.g. 'train', 'dev', 'train.masked', 'dev.masked'). The values
        of these dictionaries are outlined below:

            - processed_dataset: A dictionary, keyed by example id, containing all mentions (a
                candidate that appeared in one or more supporting documents) and their coreferents,
                a query and a dictionary mapping candidates to mention indices for each example in a
                partition. E.g. `processed_dataset['train']` looks like

                'example_id': {
                    'mentions': [
                        {
                            'text' : Mention text as it appears in supporting doc.
                            'corefs': A list of coreferent mentions.
                        }
                        ...
                    ]
                    'query': A query, as it appears in Med- or WikiHop.
                    'candidate_indices': A dictionary, mapping the candidate to its index in
                                         `encoded_mentions`.
                }

            - encoded_mentions: A torch Tensor containing encodings for each mention. Encodings
                are produced by summing the contexualized word embeddings assigned by `model`.
            - encoded_mentions_split_sizes: A torch Tensor containing the chunk sizes. To be used
                with `torch.split()` to break `encoded_mentions` up into individual training
                examples.
            - targets: A torch Tensor containing one-hot encoded answers.
            - targets_split_sizes: A torch Tensor containing chunk sizes. To be used with
                `torch.split()` to break `targets` up into individual training examples

        Args:
            dataset (dict): A dictionary keyed by partitions ('train', 'dev', 'train.masked',
                'dev.masked') representing a loaded Wiki- or MedHop dataset.
            model (Torch.nn): Any model that defines a `predict_on_tokens()` method, which accepts
                a list containining a tokenized sentence and returns a contextualized embedding for
                each token.

        Returns:
            A 5-tuple of dictionaries, keyed by partition, containing everything we need for graph
            construction and training.
        """
        processed_dataset = {}
        encoded_mentions = {}
        encoded_mentions_split_sizes = {}
        targets = {}
        targets_split_sizes = {}

        for partition, training_examples in dataset.items():
            if training_examples:
                print("Processing partition: '{}'...".format(partition))

                processed_dataset[partition] = {}

                encoded_mentions[partition] = []
                encoded_mentions_split_sizes[partition] = []
                targets[partition] = []
                targets_split_sizes[partition] = []

                for example in tqdm(training_examples, dynamic_ncols=True):

                    example_id = example['id']
                    query = example['query'].lower()
                    answer = example['answer'].lower()
                    candidates = example['candidates']

                    processed_dataset[partition][example_id] = {
                        'mentions': [],
                        'query': query,
                        'candidate_indices': {candidate.lower(): [] for candidate in candidates},
                    }

                    # One-hot encoding of answer
                    target = [1 if candidate == answer else 0 for candidate in candidates]
                    targets[partition].append(torch.tensor(target))
                    targets_split_sizes[partition].append(len(target))

                    # Keeps track of the mentions index in encoded_mentions
                    encoded_mention_idx = 0

                    for supporting_doc in example['supports']:
                        # Process the supporting document (sentence segmentation and tokenization)
                        doc = self.nlp(supporting_doc)
                        tokens, offsets, corefs, embeddings = self._process_doc(doc, model)

                        # Get character offsets of matched candidates in supporting docs
                        candidate_offsets = self._find_candidates(candidates, supporting_doc)

                        # Returns the final data structures for mentions and their encodings
                        (processed_candidates, candidate_indices, encoded_mentions_,
                         encoded_mention_idx) = self._process_candidates(
                             candidate_offsets=candidate_offsets,
                             supporting_doc=supporting_doc,
                             tokens=tokens,
                             offsets=offsets,
                             corefs=corefs,
                             embeddings=embeddings,
                             candidate_indices=processed_dataset[partition][example_id]['candidate_indices'],
                             encoded_mention_idx=encoded_mention_idx
                            )

                        processed_dataset[partition][example_id]['mentions'].append(
                            processed_candidates
                        )
                        processed_dataset[partition][example_id]['candidate_indices'] = \
                            candidate_indices
                        encoded_mentions[partition].extend(encoded_mentions_)

                    # Keep track of encoded_mentions chunk sizes
                    encoded_mentions_split_sizes[partition].append(encoded_mention_idx)

                # Return encoded_mentions and targets as tensors
                encoded_mentions[partition] = torch.cat(encoded_mentions[partition])
                targets[partition] = torch.cat(targets[partition])

        return (processed_dataset, encoded_mentions, encoded_mentions_split_sizes, targets,
                targets_split_sizes)

    def _process_doc(self, doc, model):
        """Returns 4-tuple of tokens, offsets, corefs, and embeddings for the tokens in a given
        SpaCy `doc` object.

        Args:
            doc (spacy.Doc): SpaCy doc object containing the document to process.
            model (Torch.nn): Any model that defines a `predict_on_tokens()` method, which accepts
                a list containining a tokenized sentence and returns an embedding for each token.

        Returns:
            4-tuple of tokens, character offsets, coreference resolutions and embeddings for a given
            SpaCy `doc` object.
        """
        tokens, offsets = [], []
        embeddings = torch.tensor([])

        for sent in doc.sents:
            tokens.append([])
            # Collect text and character offsets for each token
            for token in sent:
                token_text = token.text.lower()
                tokens[-1].append(token_text)
                offsets.append((token.idx, token.idx + len(token.text)))

        # Use model to get embeddings for each token across all sents
        _, embedded_sents, orig_to_bert_tok_map = model.predict_on_tokens(tokens)

        for embedded_sent, tok_map in zip(embedded_sents, orig_to_bert_tok_map):
            # Retrieve embeddings for original tokens (drop CLS, SEP and PAD tokens)
            embedded_sent = embedded_sent.cpu().detach()
            orig_token_indices = torch.tensor(tok_map, dtype=torch.long)
            embedded_sent = torch.index_select(embedded_sent, 0, orig_token_indices)

            embeddings = torch.cat((embeddings, embedded_sent), dim=0)

        # Flatten tokens to 1D list
        tokens = list(chain.from_iterable(tokens))

        # Resolve coreferences
        corefs = [[(mention.start_char, mention.end_char) for mention in cluster]
                  for cluster in doc._.coref_clusters]

        return tokens, offsets, corefs, embeddings

    def _process_candidates(self, candidate_offsets, supporting_doc, tokens, offsets, corefs,
                            embeddings, candidate_indices, encoded_mention_idx=0):
        """Returns a three-tuple of processed candidates, encoded mentions and candidate indices.

        Returns a three-tuple of processed candidates, encoded mentions and candidate indices:

        - processed_candidates: A list containing all mentions (a candidate that appeared in the
            `supporting_doc`).
        - candidate_indices: Dictionary mapping candidates to indices in `encoded_mentions`.
        - encoded_mentions: A torch Tensor containing encodings for each mention. Encodings
            are produced by summing the word embeddings `embeddings` assigned to each
            token in a mention.
         - encoded_mention_idx (int): TODO (John).

        Args:
            candidate_offsets (list): List of tuples containing the start and end character offsets
                for candidates in `supporting_doc`.
            supporting_doc (str): A single supporting document from Wiki- or MedHop.
            tokens (list): List containing all tokens in `supporting_doc`.
            offsets (list): List of tuples containing the start and end character offsets for all
                tokens in `supporting_doc`.
            corefs (list): List of lists of tuples containing the start and end character offsets
                for all mentions in a coreference cluster.
            embeddings (torch.tensor): An tensor containing contexualized embeddings for each token
                in `tokens`.
            candidate_indices (dict): TODO (John).
            encoded_mention_idx

        Returns:
            A list of dictionaries containing 'mention', the text of a candidate found in
            a `supporting_doc`, `embeddings`, the embedding(s) for a candidate found in
            `supporting_doc` and `corefs`, a list of coreference mentions for a candidate found in
            `supporting_doc`.
        """
        processed_candidates = []
        encoded_mentions = []

        def _process_mention(start, end):
            # Get character offsets for a given mention
            mention_offsets = offsets[start:end + 1]

            # Get text as it appears in the supporting document for a given mention
            # .lower() because candidates are lowercase in Wiki- and MedHop.
            mention_text = supporting_doc[mention_offsets[0][0]: mention_offsets[-1][-1]].lower()

            # Sum the embeddings assigned by a model for the given mention to produce its encoding
            encoded_mention = torch.sum(embeddings[start: end + 1, :], dim=0).unsqueeze(0)

            # TODO: This will cause us to miss coreference mentions that don't match up perfectly
            # with candidates. Consider fuzzy string matching?
            mention_corefs = [coref for coref in corefs if
                              (mention_offsets[0][0], mention_offsets[-1][-1]) in coref]
            mention_corefs = list(chain.from_iterable(mention_corefs))

            return mention_text, mention_corefs, encoded_mention

        for cand_start, cand_end in candidate_offsets:
            # Find the start and end char offsets for a candidate, AKA a mention
            mention_start = [i for i, offset in enumerate(offsets) if offset[0] == cand_start]
            mention_end = [i for i, offset in enumerate(offsets) if offset[-1] == cand_end]

            # When candidate matches only start OR end of a token (not both), we should not process
            # this candidate as it likely represents an error in candidate list of Wiki- or MedHop
            if mention_start and mention_end:
                # Gets the mentions text, its corefs, and updates encoded_mentions
                mention_text, mention_corefs, encoded_mention = \
                    _process_mention(mention_start[0], mention_end[0])
                processed_candidates.append({'text': mention_text, 'corefs': []})
                encoded_mentions.append(encoded_mention)

                # Maintains a mapping from candidates to their position in encoded_mentions
                candidate_indices[mention_text].append(encoded_mention_idx)
                encoded_mention_idx += 1

                # The first item is the mention itself, so skip it
                for coref_start, coref_end in mention_corefs[1:]:
                    start = [i for i, offset in enumerate(offsets) if offset[0] == coref_start][0]
                    end = [i for i, offset in enumerate(offsets) if offset[-1] == coref_end][0]

                    coref_text, _, encoded_mention = _process_mention(start, end)
                    processed_candidates[-1]['corefs'].append({'text': coref_text})
                    encoded_mentions.append(encoded_mention)

                    candidate_indices[mention_text].append(encoded_mention_idx)
                    encoded_mention_idx += 1

        return processed_candidates, candidate_indices, encoded_mentions, encoded_mention_idx

    def _find_candidates(self, candidates, supporting_doc):
        """Finds all non-overlapping, case-insensitive matches of `candidates` in `supporting_doc`.

        Args:
            candidates (list): A list of strings containing candidates from Wiki- or MedHop.
            supporting_doc (str): A single supporting document from Wiki- or MedHop.

        Returns:
            List of tuples containing the start and end character offsets for all non-overlapping
            mentions of `candidates` in `supporting_doc`.
        """
        # Escape all special regex characters in candidates, group by word boundary
        candidates_regex = r'(\b|$)|(\b|^)'.join([re.escape(candidate) for candidate in candidates])
        # Matches begin and end at word boundary or beginning/end of string respectively
        pattern = re.compile(r'(\b|^){}(\b|$)'.format(candidates_regex), re.IGNORECASE)
        matches = [match.span() for match in pattern.finditer(supporting_doc)]

        return matches
