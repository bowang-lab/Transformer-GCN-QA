import argparse
import pickle

import torch

from ..preprocessor_graph import BuildGraph
from ..utils.generic_utils import make_dir


def main(input_directory, output_directory):
    """Creates a graph set given the preprocessed Wiki- or MedHop data whose path is
    given by `input_directory`. The resulting combined graph tensors and index
    dictionary are saved in `output_directory`.
    """

    samples = pickle.load(open(input_directory, 'rb'))
    buildGraph = BuildGraph(samples)
    graphs, idxs = buildGraph.build()

    # make output directory if it does not exist
    make_dir(output_directory)

    torch.save(graphs, output_directory + 'graphs.t')
    pickle.dump(idxs, open(output_directory + 'sample_indices.pickle', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Builds the required graphs for each sample'
                                                  ' given the dictionaries obtained from'
                                                  ' `preprocess_wikihop`. Saves results to disk.'))
    parser.add_argument('-i', '--input', help=('Path to the preprocessed Wiki- or MedHop'
                                               ' dataset dictionary pickle.'))
    parser.add_argument('-o', '--output', help='Path to save the output graphs.')

    args = parser.parse_args()

    # TODO: type check input to ensure it is a pickle.

    main(args.input, args.output)
