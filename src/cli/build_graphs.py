import argparse
from glob import glob
import json
import os

import torch

from ..graph_builder import GraphBuilder

# TODO (Duncan): It might be nicer if GraphBuilder took samples as a dictionary keyed by partition,
# and returned a dictionary containing all graphs and chunk sizes keyed by partition. Then there
# would be one call to GraphBuilder, followed by one loop over partitions to save files.


def main(**kwargs):
    """Builds a set of graphs given the preprocessed Wiki- or MedHop dataset at `input_directory`.

    Given a preprocessed Wiki- or MedHop dataset at `input_directory`, creates a graphs for each
    partition in `input_directory/<partition>`. Maintains a list of chunk sizes that can be used
    with `torch.split()` to get a tuple of graphs, one per training example. The graphs and chunk
    sizes for each dataset partition are saved under `input_directory/<partition>`.

    Args:
        input_directory (str): Path to the preprocessed Wiki- or MedHop dataset.
    """
    partitions = glob(os.path.join(kwargs['input'], '*'))

    for partition_filepath in partitions:

        partition = os.path.basename(partition_filepath)
        print("Processing partition: '{}'...".format(partition))

        # Load the only file we need for graph building
        processed_dataset_filepath = os.path.join(partition_filepath, 'processed_dataset.json')

        with open(processed_dataset_filepath, 'r') as f:
            processed_dataset = json.load(f)

        # Build the graphs
        graph_builder = GraphBuilder(processed_dataset)
        graphs, graph_split_sizes = graph_builder.build(complement=kwargs['complement'])

        dropped = sum([1 for graph in graph_split_sizes if graph == 0])
        n_graphs = len(graph_split_sizes)

        print(f'Found {dropped}/{n_graphs} ({(dropped / n_graphs):.2%}) empty graphs.')

        # Save graphs and their chunk sizes
        graphs_filepath = os.path.join(partition_filepath, 'graphs.pt')
        graph_split_sizes_filepath = os.path.join(partition_filepath, 'graph_split_sizes.json')

        torch.save(graphs, graphs_filepath)

        with open(graph_split_sizes_filepath, 'w') as f:
            json.dump(graph_split_sizes, f, indent=2)


if __name__ == '__main__':
    description = '''Builds a set of graphs for the given preprocessed Wiki- or MedHop dataset at
    `input_directory`.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the preprocessed Wiki- or MedHop dataset.')
    parser.add_argument('--complement', action='store_true',
                        help=('Optional, provide this argument if COMPLEMENT edges should be'
                              ' added when building the graph'))

    args = vars(parser.parse_args())

    # TODO: type check input to ensure it is a pickle.

    main(**args)
