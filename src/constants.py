# The spaCy model to load. Defaults to the large sized English model.
SPACY_MODEL = 'en_core_web_lg'

# Greedyness of NeuralCoref. See here: https://github.com/huggingface/neuralcoref#parameters
NEURALCOREF_GREEDYNESS = 0.40

# Pre-trained BERT models. See here: https://github.com/huggingface/pytorch-pretrained-BERT
# for more information.
PRETRAINED_BERT_MODELS = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
]
# The default pre-trained BERT model to use
PRETRAINED_BERT_MODEL = 'bert-base-uncased'

PAD = '[PAD]'  # Special token representing a sequence pad
CLS = '[CLS]'  # Special BERT classification token
SEP = '[SEP]'  # Special BERT sequence seperator token

# Maximum size of a training example, and examples exceeding this size are skipped.
TRAIN_SIZE_THRESHOLD = 10000
