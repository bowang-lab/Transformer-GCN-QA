import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TransformerGCNQA",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    license="MIT",
    description="End-to-end neural question answering achitecture based on transformers and GCNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/berc-uoft/Transformer-GCN-QA",
    python_requires='>=3',
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Operating System :: OS Independent",
    ],
    keywords=[
        'Natural Language Processing',
        'Question Answering',
        'Transformers',
        'Graph Convolutional Neural Networks'
        'BERT',
        'Multi-hop Question Answering',
    ],
    install_requires=[
        "spacy>=2.1.3",
        # SpaCy language model, straight from GitHub
        "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz",
        "neuralcoref>=4.0",
        "pytorch-pretrained-bert>=0.6.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "tox",
            "tb-nightly",  # Until 1.14 moves to the release channel"
        ]
    },
    zip_safe=False,
)
