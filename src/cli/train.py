"""A script for training the TransformerGCNQA model.

Usage:
    `python -m src.cli.train -i ./path/to/preprocessed/dataset`
"""
import argparse
from pprint import pprint

from torch.optim import Adam

from ..models import TransformerGCNQA
from ..utils.dataset_utils import get_dataloaders
from ..utils.dataset_utils import load_preprocessed_wikihop
from ..utils.train_utils import train
from ..utils.train_utils import warn_about_big_graphs


def main(**kwargs):
    """Runs a training loop for the `TransformerGCNQA` model.
    """
    # Load preprocessed Wiki- or MedHop dataset
    processed_dataset, encoded_mentions, graphs, targets = \
        load_preprocessed_wikihop(kwargs['input'])

    dataloaders = get_dataloaders(processed_dataset, encoded_mentions, graphs, targets)

    # Warn user with number of empty and big graphs
    warn_about_big_graphs(dataloaders)

    model = TransformerGCNQA(**kwargs)

    optimizer = Adam(model.parameters(), lr=kwargs['learning_rate'])

    train(model, optimizer, processed_dataset, dataloaders, **kwargs)


if __name__ == '__main__':
    description = ''''Runs a training loop for the TransformerGCNQA model.'''
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the preprocessed Wiki- or MedHop dataset.')
    # Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int, required=False,
                        help=('Optional, effective batch size achieved with gradient accumulation.'
                              ' Defaults to 32.'))
    parser.add_argument('--dropout_rate', default=0.1, type=float, required=False,
                        help='Optional, dropout rate. Defaults to 0.1.')
    parser.add_argument('--epochs', default=20, type=int, required=False,
                        help='Optional, number of epochs to train the model for. Defaults to 20.')
    parser.add_argument('--grad_norm', default=1.0, type=float, required=False,
                        help=('Optional, maximum L2 norm to clip all parameter gradients. Set to'
                              '0 to turn off. Defaults to 1'))
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, required=False,
                        help='Optional, learning rate for the optimizer. Defaults to 1e-4.')
    # R-GCN
    parser.add_argument('--n_rgcn_layers', default=3, type=int, required=False,
                        help='Optional, number of layers in the R-GCN. Defaults to 3.')
    parser.add_argument('--n_rels', default=3, type=int, required=False,
                        help=('Optional, number of relations in the R-GCN. Set this to 4 if graphs'
                              ' were built with complement=True, otherwise set to 3. Defaults to'
                              ' 3.'))
    parser.add_argument('--rgcn_size', default=512, type=int, required=False,
                        help='Optional, dimensionality of the R-GCN layers. Defaults to 512.')
    parser.add_argument('--n_rgcn_bases', default=2, type=int, required=False,
                        help='TODO (Duncan).')
    # Other
    parser.add_argument('--evaluation_step', default=1, type=int, required=False,
                        help=('Optional, evaluate model on every evaluation_step number of epochs.'
                              'E.g. if evaluation_step==2, model is evaluated on every 2nd epoch'
                              ' Defaults to 1.'))
    parser.add_argument('--verbose', action='store_true')

    kwargs = vars(parser.parse_args())

    if kwargs['verbose']:
        pprint(kwargs)

    main(**kwargs)
