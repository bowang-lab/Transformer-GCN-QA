import json
import os
from glob import glob

import torch

# TODO (John): Rewrite this to use googledrivedownloader


def load_wikihop(directory, load_masked=False):
    """Loads the Wiki- or MedHop dataset given at `directory`.

    Loads the Wiki- or MedHop dataset given at `directory` and returns a dictionary keyed by
    partition names: 'train', 'dev', 'train.masked', 'dev.masked'. If `load_masked`, the masked
    partitions of the dataset are loaded, otherwise, 'train.masked' and 'dev.masked' in the returned
    dictionary are None.

    Args:
        directory (str): Directory path to the Wiki- or MedHop datasets.
        load_masked (bool): True if the 'train.masked' and 'dev.masked' partitions of the dataset
            should be loaded. Defaults to False.

    Returns:
        A dictionary representing the WikiHop or MedHop dataset at `directory`, keyed by partition
        names: 'train', 'dev', 'train.masked', 'dev.masked'.

    References:
        - https://qangaroo.cs.ucl.ac.uk/
    """
    dataset = {'train': None, 'dev': None, 'train.masked': None, 'dev.masked': None}

    for file in os.listdir(directory):

        filename = os.fsdecode(file)
        partition = filename.split('.json')[0]
        filepath = os.path.join(directory, filename)

        if filename.endswith('.json'):
            # Only load masked partitions if `load_masked` is True
            if 'masked' not in partition or load_masked:
                with open(filepath, 'r') as f:
                    dataset[partition] = json.loads(f.read())

    return dataset


def load_preprocessed_wikihop(directory):
    """Loads a preprocessed Wiki- or MedHop dataset given at `directory`.

    Loads the preprocessed Wiki- or MedHop dataset given at `directory` and returns 4 dictionaries
    keyed by partition names, e.g. 'train', 'dev', 'train.masked', 'dev.masked', containing to
    the `processed_dataset`, the `encoded_mentions`, `graphs` and `targets`.

    TODO (John): Breakdown of what each dictionary contains.

    Args:
        directory (str): Directory path to the preprocessed Wiki- or MedHop datasets.

    Returns:
        Four dictionaries, keyed by dataset partitions, containing everything we need to train the
        model.
    """
    processed_dataset = {}
    encoded_mentions = {}
    graphs = {}
    targets = {}

    partitions = glob(os.path.join(directory, '*'))

    for partition_filepath in partitions:

        partition = os.path.basename(partition_filepath)

        processed_dataset_filepath = os.path.join(partition_filepath, 'processed_dataset.json')
        encoded_mentions_filepath = os.path.join(partition_filepath, 'encoded_mentions.pt')
        encoded_mentions_split_sizes_filepath = os.path.join(partition_filepath,
                                                             'encoded_mentions_split_sizes.json')
        graphs_filepath = os.path.join(partition_filepath, 'graphs.pt')
        graph_split_sizes_filepath = os.path.join(partition_filepath, 'graph_split_sizes.json')
        targets_filepath = os.path.join(partition_filepath, 'targets.pt')
        targets_split_sizes_filepath = os.path.join(partition_filepath, 'targets_split_sizes.json')

        # Load .json files
        with open(processed_dataset_filepath, 'r') as f:
            processed_dataset[partition] = json.load(f)
        with open(encoded_mentions_split_sizes_filepath, 'r') as f:
            encoded_mentions_split_sizes = json.load(f)
        with open(graph_split_sizes_filepath, 'r') as f:
            graph_split_sizes = json.load(f)
        with open(targets_split_sizes_filepath, 'r') as f:
            targets_split_sizes = json.load(f)

        encoded_mentions[partition] = \
            torch.split(torch.load(encoded_mentions_filepath), encoded_mentions_split_sizes)
        graphs[partition] = torch.split(torch.load(graphs_filepath), graph_split_sizes)
        targets[partition] = torch.split(torch.load(targets_filepath), targets_split_sizes)

    return processed_dataset, encoded_mentions, graphs, targets
