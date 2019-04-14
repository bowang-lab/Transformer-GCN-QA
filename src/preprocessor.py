import neuralcoref
import spacy
from tqdm import tqdm


class Preprocessor():
    """This class provides methods for processing raw text. Most importantly,
    `Preprocessor.transform()` can be used to transform the Wiki- or MedHop datasets into a format
    that can then be used to construct a graph for learning with a GCN.
    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        # adds coref to spaCy pipeline with medium sized english language model
        neuralcoref.add_to_pipe(self.nlp)

    def transform(self, dataset, model=None):
        """
        Modifies given Wiki- or MedHop `dataset` to contain everything we need for graph construction.

        For the given Wiki- or MedHop `dataset`, modifies the object to contain everything we need
        for graph construction. Namely, for each training example in `dataset`, the key
        'processed_supports' is added, containing a 3-tuple of tokens, embeddings and named entity
        labels. Labels and embeddings are derived from `model`, which is expected to be a PyTorch
        implementation of a BERT token classifier. If `model` is None, embeddings and labels are
        provided by a SpaCy language model.

        Args:
            dataset (dict): A dictionary keyed by partitions ('train', 'dev', 'train.masked',
                'dev.masked') representing a loaded Wiki- or MedHop dataset.
            model (Torch.nn): A PyTorch implementation of a BERT token classification model used to
                extract embeddings and named entity labels. If None, embeddings and labels are
                provided by a SpaCy language model. Defaults to None.

        Returns:
            `dataset`, where each training example has the added key 'processed_supports', pointing
            to a list of tuples containing tokens, embeddings and named entity labels for each
            supporting document for the training example.
        """
        for partition, training_examples in dataset.items():
            # partition is None, if not provided
            if partition:
                print('Processing partition: {}...'.format(partition))
                for example in tqdm(training_examples):
                    for supporting_doc in example['supports']:
                        # process the supporting document (sentence segmenation and tokenization)
                        doc = self.nlp(supporting_doc)
                        tokens, embeddings, labels, _ = self._process_doc(doc)

                        # add list of lists of tuples, representing the sentences and tokens for each
                        # supporting document, where the tuples contain the (text, label, and embedding)
                        if 'processed_supports' not in example:
                            example['processed_supports'] = []
                        example['processed_supports'].append((tokens, embeddings, labels))

        return dataset

    def _process_doc(self, doc):
        """Returns tuple of tokens, embeddings, named entity labels and character offsets for the
        tokens in a given in SpaCy `doc` object.
        """
        # accumulators
        tokens, embeddings, labels, offsets = [], [], [], []

        # collect token sequence
        for sent in doc.sents:
            tokens.append([])
            embeddings.append([])
            labels.append([])
            offsets.append([])

            for token in sent:
                tokens[-1].append(token.text)
                embeddings[-1].append(token.vector)
                labels[-1].append(token.ent_type_)
                offsets[-1].append((token.idx, token.idx + len(token.text)))

        return tokens, embeddings, labels, offsets
