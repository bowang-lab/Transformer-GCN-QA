import argparse

from torch.optim import Adam

from ..models import TransformerGCNQA
from ..utils.dataset_utils import get_dataloaders
from ..utils.dataset_utils import load_preprocessed_wikihop
from ..utils.model_utils import train


def main(**kwargs):
    """Runs a training loop for the `TransformerGCNQA` model.

    Args:
        input_directory (str): Path to the preprocessed Wiki- or MedHop dataset.
    """
    model = TransformerGCNQA(**kwargs)

    optimizer = Adam(model.parameters(), lr=kwargs['learning_rate'])

    # Load preprocessed Wiki- or MedHop dataset
    processed_dataset, encoded_mentions, graphs, targets = \
        load_preprocessed_wikihop(kwargs['input'])

    dataloaders = get_dataloaders(processed_dataset, encoded_mentions, graphs, targets)

    train(model, optimizer, processed_dataset, dataloaders, **kwargs)


if __name__ == '__main__':
    description = ''''Runs a training loop for the TransformerGCNQA model.'''
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the preprocessed Wiki- or MedHop dataset.')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, required=False,
                        help=('Optional, effective batch size achieved with gradient accumulation.'
                              ' Defaults to 32.'))
    parser.add_argument('-dr', '--dropout_rate', default=0.2, type=float, required=False,
                        help='Optional, dropout rate. Defaults to 0.2.')
    parser.add_argument('-ep', '--epochs', default=20, type=int, required=False,
                        help='Optional, number of epochs to train the model for. Defaults to 20.')
    parser.add_argument('-gn', '--grad_norm', default=1.0, type=float, required=False,
                        help=('Optional, maximum norm to clip all parameter gradients. Defaults to'
                              ' 1.0.'))
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, required=False,
                        help='Optional, learning rate for the optimizer. Defaults to 1e-4.')

    args = vars(parser.parse_args())

    main(**args)
