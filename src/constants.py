# The spaCy model to load. Defaults to the large sized English model.
SPACY_MODEL = 'en_core_web_lg'

# Greedyness of NeuralCoref. See here: https://github.com/huggingface/neuralcoref#parameters
NEURALCOREF_GREEDYNESS = 0.10

# The default pre-trained BERT model to use
PRETRAINED_BERT_MODEL = 'bert-base-uncased'

PAD = '[PAD]'  # Special token representing a sequence pad
CLS = '[CLS]'  # Special BERT classification token
SEP = '[SEP]'  # Special BERT sequence seperator token

# Maximum size of a training example, and examples exceeding this size are skipped.
TRAIN_SIZE_THRESHOLD = 3000
EVAL_SIZE_THRESHOLD = 3500
