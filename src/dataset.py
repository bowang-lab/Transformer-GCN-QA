from torch.utils import data


class Dataset(data.Dataset):
    """TODO (John).
    """
    def __init__(self, encoded_mentions, graphs, targets):
        self.encoded_mentions = encoded_mentions
        self.graphs = graphs
        self.targets = targets

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.targets)

    def __getitem__(self, index):
        """Generates one sample of data"""
        encoded_mentions = self.encoded_mentions[index]
        graph = self.graphs[index]
        target = self.targets[index]

        return index, encoded_mentions, graph, target
