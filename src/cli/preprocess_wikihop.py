import argparse
import json
import os
import pickle

from ..models import BERT
from ..preprocessor import Preprocessor
from ..utils.dataset_utils import load_wikihop
from ..utils.generic_utils import make_dir


def main(input_directory, output_directory):
    """Creates a dictionary from the given Wiki- or MedHop dataset (given at `input_directory`)
    which contains everything we need for graph construction. Saves the resulting dataset to 
    `output_directory`.
    """
    dataset = load_wikihop(input_directory)
    preprocessor = Preprocessor()
    model = BERT()

    processed_dataset = preprocessor.transform(dataset, model)

    # make output directory if it does not exist
    make_dir(output_directory)

    pickle.dump(processed_dataset, open(output_directory + "processed_dataset.pickle", "wb" ))

    return processed_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Creates a dictionary for the given Wiki- or'
                                                  ' MedHop dataset which contains everything we'
                                                  ' need for graph construction. Saves the'
                                                  ' resulting dataset to disk.'))
    parser.add_argument('-i', '--input', help='Path to the Wiki- or MedHop dataset.')
    parser.add_argument('-o', '--output', help=('Path to save the processed output for the Wiki-'
                                                ' or MedHop dataset.'))

    args = parser.parse_args()

    main(args.input, args.output)
