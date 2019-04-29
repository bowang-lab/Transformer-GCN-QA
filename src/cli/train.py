import argparse

from torch.optim import Adam

from ..models import TransformerGCNQA
from ..utils.dataset_utils import get_dataloaders
from ..utils.dataset_utils import load_preprocessed_wikihop
from ..utils.model_utils import train


def main(input_directory, **kwargs):
    """Runs a training loop for the `TransformerGCNQA` model.

    Args:
        input_directory (str): Path to the preprocessed Wiki- or MedHop dataset.
    """
    model = TransformerGCNQA()

    optimizer = Adam(model.parameters())

    # Load preprocessed Wiki- or MedHop dataset
    processed_dataset, encoded_mentions, graphs, targets = \
        load_preprocessed_wikihop(input_directory)

    dataloaders = get_dataloaders(processed_dataset, encoded_mentions, graphs, targets)

    train(model, optimizer, processed_dataset, dataloaders, **kwargs)


if __name__ == '__main__':
    description = ''''Runs a training loop for the TransformerGCNQA model.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', help='Path to the preprocessed Wiki- or MedHop dataset.')

    # parser.add_argument('-d', '--dropout', help='Dropout rate.')
    parser.add_argument('-e', '--epochs', default=20, type=int, required=False,
                        help='Optional, number of epochs to train the model for. Defaults to 20.')

    args = parser.parse_args()
    main(args.input, epochs=args.epochs)
