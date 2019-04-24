import re
from itertools import chain

import neuralcoref
import spacy
import torch
from tqdm import tqdm

from .constants import SPACY_MODEL


class Preprocessor(object):
    """This class provides methods for processing the Wiki- and MedHop datasets. Most importantly,
    `Preprocessor.transform()` can be used to transform the Wiki- or MedHop datasets into a format
    that can then be used to construct a graph for learning with a GCN.

    Attributes:
        nlp (spacy.lang): Optional, SpaCy language model. If None, loads `constants.SPACY_MODEL`.
            Defaults to None.
    """
    def __init__(self, nlp=None):
        # SpaCy object for processing natural language
        self.nlp = nlp if nlp else spacy.load(SPACY_MODEL)
        # Adds coref to spaCy pipeline with medium sized english language model
        neuralcoref.add_to_pipe(self.nlp)

    def transform(self, dataset, model=None):
        """Extracts from given Wiki- or MedHop `dataset` everything we need for graph construction.

        For the given Wiki- or MedHop `dataset`, returns a two-tuple containing a dictionary with
        everything we need for graph construction along with a torch Tensor containing encodings
        for each mention (a candidate that appeared in one or more supporting documents). Encodings
        are produced by summing the contexualized word embeddings assigned by `model`.

        The dictionaries structure looks like:

        'partition' : {
            'training_example_id': [
                { 'mention' : The mention text as it appeared in the supporting doc
                  'encoding_idx' : An index into `encoded_mentions`
                  'corefs': A list of coreferring mentions
                }
                ...
            ]
        }

        Args:
            dataset (dict): A dictionary keyed by partitions ('train', 'dev', 'train.masked',
                'dev.masked') representing a loaded Wiki- or MedHop dataset.
            model (Torch.nn): Any model that defines a `predict_on_tokens()` method, which accepts
                a list containining a tokenized sentence and returns a contextualized embedding for
                each token.

        Returns:
            A two-tuple containing a dictionary with everything we for graph construction along with
            a torch Tensor containing contexualized word embeddings.
        """
        processed_dataset = {}
        encoded_mentions = []

        for partition, training_examples in dataset.items():
            if training_examples:
                print("Processing partition: '{}'...".format(partition))

                processed_dataset[partition] = {}

                for example in tqdm(training_examples):

                    id_ = example['id']
                    candidates = example['candidates']

                    processed_dataset[partition][id_] = []

                    for supporting_doc in example['supports']:
                        # Process the supporting document (sentence segmentation and tokenization)
                        doc = self.nlp(supporting_doc)
                        tokens, offsets, corefs, embeddings = self._process_doc(doc, model)

                        # Get character offsets of matched candidates in supporting docs
                        candidate_offsets = self._find_candidates(candidates, supporting_doc)

                        # Returns the final data structures for mentions and their encodings
                        processed_candidates, encoded_mentions_ = \
                            self._process_candidates(candidate_offsets=candidate_offsets,
                                                     supporting_doc=supporting_doc,
                                                     tokens=tokens,
                                                     offsets=offsets,
                                                     corefs=corefs,
                                                     embeddings=embeddings)

                        processed_dataset[partition][id_].append(processed_candidates)
                        encoded_mentions.extend(encoded_mentions_)

        encoded_mentions = torch.cat(encoded_mentions)

        return processed_dataset, encoded_mentions

    def _process_doc(self, doc, model):
        """Returns 4-tuple of tokens, character offsets, embeddings, coreference resolutions and the
        tokens in a given in SpaCy `doc` object.

        Args:
            doc (spacy.Doc): SpaCy doc object containing the document to process.
            model (Torch.nn): Any model that defines a `predict_on_tokens()` method, which accepts
                a list containining a tokenized sentence and returns an embedding for each token.

        Returns:
            4-tuple of tokens, embeddings, coreference resolutions and character offsets for a given
            in SpaCy `doc` object.
        """
        tokens, offsets = [], []
        contextualized_embeddings = torch.tensor([])

        for sent in doc.sents:
            tokens.append([])
            # Collect text and character offsets for each token
            for token in sent:
                tokens[-1].append(token.text)
                offsets.append((token.idx, token.idx + len(token.text)))

        # Use model to get embeddings for each token across all sents
        embedded_docs, orig_to_bert_tok_map = model.predict_on_tokens(tokens)

        for embedded_doc, tok_map in zip(embedded_docs, orig_to_bert_tok_map):
            # Retrieve embeddings for original tokens (drop CLS, SEP and PAD tokens)
            embedded_doc = embedded_doc.cpu().detach()
            orig_token_indices = torch.tensor(tok_map, dtype=torch.long)
            embedded_doc = torch.index_select(embedded_doc, 0, orig_token_indices)

            contextualized_embeddings = torch.cat((contextualized_embeddings, embedded_doc), dim=0)

        # Flatten tokens to 1D list
        tokens = list(chain.from_iterable(tokens))

        # Resolve coreferences
        corefs = [[(mention.start_char, mention.end_char) for mention in cluster]
                  for cluster in doc._.coref_clusters]

        return tokens, offsets, corefs, contextualized_embeddings

    def _process_candidates(self, candidate_offsets, supporting_doc, tokens, offsets, corefs,
                            embeddings, encoded_mention_idx=0):
        """Returns a list of dictionaries representing processed candidate mentions.

        Returns a list of dictionaries representing processed candidate mentions. Candidates are
        found by searching `tokens` for tokens with character offsets matching those in
        `candidate_offsets`. For every mention, a dictionary containing the mention along with its
        corresponding, contextualized embedding from `embeddings` and all coreference mentions.

        Args:
            candidate_offsets (list): List of tuples containing the start and end character offsets
                for candidates in `supporting_doc`.
            supporting_doc (str): A single supporting document from Wiki- or MedHop.
            tokens (list): List containing all tokens in `supporting_doc`.
            embeddings (torch.tensor): An tensor containing contexualized embeddings for each token
                in `tokens`.
            corefs (list): List of lists of tuples containing the start and end character offsets
                for all mentions in a coreference cluster.
            offsets (list): List of tuples containing the start and end character offsets for all
                tokens in `supporting_doc`.

        Returns:
            A list of dictionaries containing 'mention', the text of a candidate found in
            a `supporting_doc`, `embeddings`, the embedding(s) for a candidate found in
            `supporting_doc` and `corefs`, a list of coreference mentions for a candidate found in
            `supporting_doc`.
        """
        processed_candidates, encoded_mentions = [], []

        def _process_mention(start, end):
            # Get character offsets for a given mention
            mention_offsets = offsets[start:end + 1]

            # Get text as it appears in the supporting document for a given mention
            mention_text = supporting_doc[mention_offsets[0][0]: mention_offsets[-1][-1]]

            # Sum the embeddings assigned by a model for the given mention to produce its encoding
            encoded_mention = torch.sum(embeddings[start: end + 1, :], dim=0).unsqueeze(0)
            encoded_mentions.append(encoded_mention)

            # TODO: This will cause us to miss coreference mentions that don't match up perfect with
            # candidates. Consider fuzzy string matching?
            mention_corefs = [coref for coref in corefs if
                              (mention_offsets[0][0], mention_offsets[-1][-1]) in coref]
            mention_corefs = list(chain.from_iterable(mention_corefs))

            return mention_text, mention_corefs

        for cand_start, cand_end in candidate_offsets:
            # Find the start and end char offsets for a candidate, AKA a mention
            mention_start = [i for i, offset in enumerate(offsets) if offset[0] == cand_start]
            mention_end = [i for i, offset in enumerate(offsets) if offset[-1] == cand_end]

            # When candidate matches only start OR end of a token (not both), we should not process
            # this candidate as it likely represents an error in candidate list of Wiki- or MedHop
            if mention_start and mention_end:
                mention_text, mention_corefs = _process_mention(mention_start[0], mention_end[0])

                processed_candidates.append({'mention': mention_text,
                                             'encoding_idx': encoded_mention_idx,
                                             'corefs': []})

                encoded_mention_idx += 1

                # The first item is the mention itself, so skip it
                for coref_start, coref_end in mention_corefs[1:]:
                    # Find the start and end char offsets for a coref
                    start = [i for i, offset in enumerate(offsets) if offset[0] == coref_start][0]
                    end = [i for i, offset in enumerate(offsets) if offset[-1] == coref_end][0]

                    coref_text, _, = _process_mention(start, end)

                    processed_candidates[-1]['corefs'].append({'mention': coref_text,
                                                               'encoding_idx': encoded_mention_idx})

                    encoded_mention_idx += 1

        return processed_candidates, encoded_mentions

    def _find_candidates(self, candidates, supporting_doc):
        """Finds all non-overlapping matches of `candidates` in `supporting_doc`.

        Args:
            candidates (list): A list of strings containing candidates from Wiki- or MedHop.
            supporting_doc (str): A single supporting document from Wiki- or MedHop.

        Returns:
            List of tuples containing the start and end character offsets for all non-overlapping
            mentions of `candidates` in `supporting_doc`.
        """
        pattern = re.compile(r'\b{}\b'.format(r'\b|\b'.join(candidates)), re.IGNORECASE)
        matches = [match.span() for match in pattern.finditer(supporting_doc)]

        return matches
