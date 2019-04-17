import json
import os

# TODO (John): Rewrite this to use googledrivedownloader

def load_wikihop(directory, load_masked=False):
    """Loads the WikiHop or MedHop dataset given at `directory`.

    Loads the WikiHop or MedHop dataset given at `directory` and returns a dictionary keyed by
    partition names: 'train', 'dev', 'train.masked', 'dev.masked'. If `load_masked`, the masked
    partitions of the dataset are loaded, otherwise, 'train.masked' and 'dev.masked' in the returned
    dictionary are None.

    Args:
        directory (str): Directory path to the WikiHop or MedHop datasets.
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
            # only load masked partitions if `load_masked` is True
            if 'masked' not in partition or load_masked:
                with open(filepath, 'r') as f:
                    dataset[partition] = json.loads(f.read())

    return dataset
