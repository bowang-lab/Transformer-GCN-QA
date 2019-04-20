# The spaCy model to load. Defaults to the medium sized English model.
SPACY_MODEL = 'en_core_web_md'
# Pre-trained BERT models. See here: https://github.com/huggingface/pytorch-pretrained-BERT 
# for more information.
PRETRAINED_BERT_MODELS= [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
]

# Special token representing a sequence pad
PAD = '[PAD]'
CLS = '[CLS]'
SEP = '[SEP]'
