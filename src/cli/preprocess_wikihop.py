import argparse
import json
import os

import torch

from ..models import BERT
from ..preprocessor import Preprocessor
from ..utils.dataset_utils import load_wikihop
from ..utils.generic_utils import make_dir


def main(input_directory, output_directory):
    """Creates a json for each partition in the the given Wiki- or MedHop dataset
    (`input_directory`) which contains everything we need for graph construction along with a
    pickled torch Tensor containing contextualized embeddings. The json files are saved to
    `output_directory/<partition>.json` and the pickle file to `output_directory/embeddings.pickle`.
    """
    dataset = load_wikihop(input_directory)
    preprocessor = Preprocessor()
    model = BERT()

    # Process the dataset, extracting what we need for graph construction
    processed_dataset, embeddings = preprocessor.transform(dataset, model)

    # Make output directory if it does not exist
    make_dir(output_directory)

    # Save processed dataset as a json file, one per partition
    for partition in processed_dataset:

        output_filepath = os.path.join(output_directory, "{}.json".format(partition))

        with open(output_filepath, 'w') as f:
            json.dump(processed_dataset[partition], f)

    # Save embeddings, which are a PyTorch Tensor
    ouput_filepath_embeddings = os.path.join(output_directory, 'embeddings.pickle')
    torch.save(ouput_filepath_embeddings)

    return processed_dataset


if __name__ == '__main__':
    description = '''Creates a json for each partition in the the given Wiki- or MedHop dataset
    (`input_directory`) which contains everything we need for graph construction along with a
    pickled torch Tensor containing contextualized embeddings. The json files are saved to
    `output_directory/<partition>.json` and the pickle file to `output_directory/embeddings.pickle`.
    '''
    parser = argparse.ArgumentParser(description=(description))
    parser.add_argument('-i', '--input', help='Path to the Wiki- or MedHop dataset.')
    parser.add_argument('-o', '--output', help=('Path to save the processed output for the Wiki-'
                                                ' or MedHop dataset.'))

    args = parser.parse_args()

    main(args.input, args.output)
