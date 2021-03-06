import json
import os
from glob import glob

import torch
from torch.utils import data

from ..dataset import Dataset

# TODO (John): Rewrite this to use googledrivedownloader


def load_wikihop(directory, masked=False):
    """Loads the Wiki- or MedHop dataset given at `directory`.

    Loads the Wiki- or MedHop dataset given at `directory` and returns a dictionary keyed by
    partition names. If `masked`, only the masked partitions of the dataset are loaded
    ('train.masked', 'dev.masked'). Otherwise, only the unmasked partitions are loaded
    ('train, 'dev').

    Args:
        directory (str): Directory path to the Wiki- or MedHop datasets.
        masked (bool): If True, only the masked partitions of the dataset are loaded
            ('train.masked', 'dev.masked'). Otherwise, only the unmasked partitions are loaded
            ('train, 'dev'). Defaults to False.

    Returns:
        A dictionary representing the WikiHop or MedHop dataset at `directory`, keyed by partition
        names.

    References:
        - https://qangaroo.cs.ucl.ac.uk/
    """
    dataset = {}

    partitions = [p for p in glob(os.path.join(directory, '*'))
                  # Only load masked partitions if `masked` is True
                  if ('.masked' in p and masked) or ('.masked' not in p and not masked)]

    for partition in partitions:
        with open(partition, 'r') as f:
            partition_key = os.path.splitext(os.path.basename(partition))[0]
            dataset[partition_key] = json.loads(f.read())

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
        graphs[partition] = torch.split(torch.load(graphs_filepath), graph_split_sizes, dim=-1)
        targets[partition] = torch.split(torch.load(targets_filepath), targets_split_sizes)

    return processed_dataset, encoded_mentions, graphs, targets


def get_dataloaders(processed_dataset, encoded_mentions, graphs, targets):
    """Gets dataloaders for given preprocessed Wiki- or MedHop dataset.

    Args:
        processed_dataset (dict): TODO, see above^!
        encoded_mentions (dict): TODO, see above^!
        graphs (dict): TODO, see above^!
        targets (dict): TODO, see above^!

    Returns:
        A dictionary containing `DataLoader` objects for each partition in `processed_dataset`.
    """
    dataloaders = {}
    for partition in processed_dataset:
        dataset = Dataset(encoded_mentions[partition], graphs[partition], targets[partition])

        shuffle = partition == 'train'
        dataloaders[partition] = data.DataLoader(dataset, shuffle=shuffle)

    return dataloaders
