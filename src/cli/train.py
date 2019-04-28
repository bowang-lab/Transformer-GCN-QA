import argparse

from torch.utils import data
from torch.optim import Adam

from ..dataset import Dataset
from ..utils.dataset_utils import load_preprocessed_wikihop
from ..utils.model_utils import train
from ..models import TransformerGCNQA


def main(input_directory):
    """Runs a training loop for the `TransformerGCNQA` model.
    """
    # Define our model
    model = TransformerGCNQA()

    # Define our optimizer
    optimizer = Adam(model.parameters())

    # Load preprocessed dataset
    processed_dataset, encoded_mentions, graphs, targets = \
        load_preprocessed_wikihop(input_directory)

    # Get the dataloaders
    dataloaders = {}
    for partition in processed_dataset:
        shuffle = True if partition == 'train' else False

        dataset = Dataset(encoded_mentions[partition], graphs[partition], targets[partition])
        dataloaders[partition] = data.DataLoader(dataset, shuffle=shuffle)

    # Train the model
    train(model, optimizer, processed_dataset, dataloaders)


if __name__ == '__main__':
    description = ''''Runs a training loop for the TransformerGCNQA model.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', help='Path to the preprocessed Wiki- or MedHop dataset.')
    # TODO: Arguments for hyperparams, consider using a config file

    args = parser.parse_args()
    main(args.input)
