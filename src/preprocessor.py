import re
import time
from itertools import chain

import neuralcoref
import spacy
import torch
from tqdm import tqdm

from constants import SPACY_MODEL


class Preprocessor():
    """This class provides methods for processing raw text. Most importantly,
    `Preprocessor.transform()` can be used to transform the Wiki- or MedHop datasets into a format
    that can then be used to construct a graph for learning with a GCN.
    """
    def __init__(self, nlp=None):
        # spaCy object for processing natural language
        self.nlp = nlp if nlp else spacy.load(SPACY_MODEL)
        # adds coref to spaCy pipeline with medium sized english language model
        neuralcoref.add_to_pipe(self.nlp)

    def transform(self, dataset, model=None):
        """
        Extracts from given Wiki- or MedHop `dataset` everything we need for graph construction.

        For the given Wiki- or MedHop `dataset`, returns a dictionary that contains everything we
        for graph construction. Namely, a dictionary of dictionaries is returned that is keyed by
        training partition (the keys in `dataset`) and the unique id of each training example. Each
        training example is associated with a list of dictionaries containing the keys 'mention',
        'embeddings' and 'corefs', corresponding to a candidate, its embeddings (as torch.Tensor),
        and a list of coreferring mentions respectively.

        Args:
            dataset (dict): A dictionary keyed by partitions ('train', 'dev', 'train.masked',
                'dev.masked') representing a loaded Wiki- or MedHop dataset.
            model (Torch.nn): Any model that defines a `predict_on_tokens()` method, which accepts
                a list containining a tokenized sentence and returns an embedding for each token.

        Returns:
            A two-tuple containing a dictionary with everything we for graph construction along with
            a torch Tensor containing contexualized word embeddings.
        """
        embeddings = torch.Tensor([])
        processed_dataset = {}

        for partition, training_examples in dataset.items():
            if training_examples:
                print('Processing partition: {}...'.format(partition))

                processed_dataset[partition] = {}

                for example in tqdm(training_examples):

                    id_ = example['id']
                    candidates = example['candidates']

                    processed_dataset[partition][id_] = []

                    for supporting_doc in example['supports']:
                        # process the supporting document (sentence segmentation and tokenization)
                        doc = self.nlp(supporting_doc)

                        tokens, offsets, corefs, embeddings, embedding_indices = \
                            self._process_doc(doc=doc,
                                              model=model,
                                              embeddings=embeddings)

                        # returns char offsets matched candidates in supporting docs
                        candidate_offsets = self._find_candidates(candidates, supporting_doc)
                        processed_candidates = \
                            self._process_candidates(candidate_offsets=candidate_offsets,
                                                     supporting_doc=supporting_doc,
                                                     tokens=tokens,
                                                     offsets=offsets,
                                                     corefs=corefs,
                                                     embedding_indices=embedding_indices)

                        processed_dataset[partition][id_].append(processed_candidates)

        return processed_dataset, embeddings

    def _process_doc(self, doc, model, embeddings):
        """Returns 4-tuple of tokens, character offsets, embeddings, coreference resolutions and  for
        the tokens in a given in SpaCy `doc` object.

        Args:
            doc (spacy.Doc): SpaCy doc object containing the document to process.
            model (Torch.nn): Any model that defines a `predict_on_tokens()` method, which accepts
                a list containining a tokenized sentence and returns an embedding for each token.

        Returns:
            4-tuple of tokens, embeddings, coreference resolutions and character offsets for a given
            in SpaCy `doc` object.
        """
        tokens, offsets, embedding_indices = [], [], []

        for sent in doc.sents:
            # Need to store the tokens seperately in order to feed them to model
            tokens_in_sent = []
            # Collect text and character offsets for each token
            for token in sent:
                tokens_in_sent.append(token.text)
                offsets.append((token.idx, token.idx + len(token.text)))

            tokens.append(tokens_in_sent)

        # Use model to get embeddings for each token across all sents
        embedded_support_doc = model.predict_on_tokens(tokens)

        for sent, embeded_doc in zip(tokens, embedded_support_doc):
            # Drop pads
            embeded_doc_ = embeded_doc[:len(sent)]

            # Track the indices of the embeddings for the given tokens
            embedding_idx_start = 0 if embeddings.shape[0] == 0 else embeddings.shape[0] - 1
            embedding_idx_end = embedding_idx_start + embeded_doc_.shape[0]

            embedding_indices_ = [i for i in range(embedding_idx_start, embedding_idx_end + 1)]
            embedding_indices.extend(embedding_indices_)

            # Add embeddings to master list
            embeddings = torch.cat((embeddings, embeded_doc_))

        # Flatten tokens to 1D list
        tokens = list(chain.from_iterable(tokens))

        # Resolve coreferences
        corefs = [[(mention.start_char, mention.end_char) for mention in cluster]
                  for cluster in doc._.coref_clusters]

        return tokens, offsets, corefs, embeddings, embedding_indices

    def _process_candidates(self, candidate_offsets, supporting_doc, tokens, offsets, corefs,
                            embedding_indices):
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
        def _process_mention(mention_start, mention_end):
            # Get character offsets for a given mention
            mention_offsets = offsets[mention_start:mention_end + 1]
            # Get text as it appears in the supporting document for a given mention
            mention_text = supporting_doc[mention_offsets[0][0]: mention_offsets[-1][-1]]
            # Get the embeddings assigned by a model for the given mention
            mention_embeddings_indices = \
                (embedding_indices[mention_start], embedding_indices[mention_end])

            # TODO: This will cause us to miss coreference mentions that don't match up perfect with
            # candidates. Consider fuzzy string matching?
            mention_corefs = [coref for coref in corefs if
                              (mention_offsets[0][0], mention_offsets[-1][-1]) in coref]
            mention_corefs = list(chain.from_iterable(mention_corefs))

            return mention_text, mention_corefs, mention_embeddings_indices

        processed_candidates = []

        for cand_start, cand_end in candidate_offsets:
            mention_start = [i for i, offset in enumerate(offsets) if offset[0] == cand_start]
            mention_end = [i for i, offset in enumerate(offsets) if offset[-1] == cand_end]

            # When candidate matches only start OR end of a token (not both), we should not process
            # this candidate as it likely represents an error in candidate list of Wiki- or MedHop
            if mention_start and mention_end:
                mention_text, mention_corefs, mention_embeddings_indices = \
                    _process_mention(mention_start[0], mention_end[0])

                processed_candidates.append({'mention': mention_text,
                                             'embeddings_indices': mention_embeddings_indices,
                                             'corefs': []})

                # the first item is the mention itself, so skip it
                for coref_start, coref_end in mention_corefs[1:]:
                    coref_start_ = [i for i, offset in enumerate(offsets) 
                                    if offset[0] == coref_start][0]
                    coref_end_ = [i for i, offset in enumerate(offsets) 
                                  if offset[-1] == coref_end][0]

                    coref_text, _, coref_embeddings_indices = \
                        _process_mention(coref_start_, coref_end_)

                    processed_candidates[-1]['corefs'].append(
                        {'mention': coref_text,
                         'embeddings_indices': coref_embeddings_indices})

        return processed_candidates

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
