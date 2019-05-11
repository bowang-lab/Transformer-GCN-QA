import torch
from ..constants import TRAIN_SIZE_THRESHOLD
from .model_utils import get_device
from tqdm import tqdm


def warn_about_big_graphs(dataloaders):
    """Prints a warning about empty or large graphs.

    Computes the percentage of empty or large graphs and prints a warning with this information.
    On the train partition, any empty or large graphs are simply dropped. On dev and test
    partitions, a random guess is made on these graphs.
    """
    for partition in dataloaders:
        big_graphs = sum([1 for _, _, graph, _ in dataloaders[partition]
                          if graph.shape[-1] == 0 or graph.shape[-1] > TRAIN_SIZE_THRESHOLD])
        perc_dropped = big_graphs / len(dataloaders[partition])

        if partition == 'train':
            print(f"Will drop {perc_dropped:.2%} of graphs from partition {partition}")
        else:
            print(f"Will randomly guess on {perc_dropped:.2%} of graphs from partition {partition}")


# TODO: There is a lot of repeated code between train / dev loops. Encapsulate
def train(model, optimizer, processed_dataset, dataloaders, **kwargs):
    """Trains an instance of `model`.

    Args:
        model (): TODO.
        optimizer (torch.optim): TODO.
        processed_dataset (dict): TODO.
        dataloaders (dict): TODO.
    """
    device, _ = get_device(model)

    # Cast to list so that we can index in
    processed_dataset = {partition: list(processed_dataset[partition].values())
                         for partition in processed_dataset}

    # Dictionary which scores per partition and per epoch performance metrics
    evaluation_scores = {partition: {epoch: None for epoch in range(1, kwargs['epochs'] + 1)}
                         for partition in processed_dataset}

    best_dev_acc = 0
    best_dev_epoch = 0

    for epoch in range(kwargs['epochs']):

        # Train loop
        model.train()

        optimizer.zero_grad()

        train_loss, train_steps = 0, 0

        pbar_descr = f"Epoch: {epoch + 1}/{kwargs['epochs']}"
        pbar = tqdm(dataloaders['train'], unit='batch', desc=pbar_descr, dynamic_ncols=True)

        batch_idx = 0

        for index, encoded_mentions, graph, target in pbar:

            index = index.item()
            encoded_mentions = encoded_mentions.to(device)
            graph = graph.to(device)
            target = target.to(device)

            # Don't train on empty graphs, or graphs to big to fit into memory
            if graph.shape[-1] == 0 or graph.shape[-1] >= TRAIN_SIZE_THRESHOLD:
                continue

            # TODO: This is kind of ugly, maybe the model itself should deal with the batch index?
            encoded_mentions = encoded_mentions.squeeze(0)
            graph = graph.squeeze(0)

            query = processed_dataset['train'][index]['query']
            candidate_indices = processed_dataset['train'][index]['candidate_indices']

            _, loss = model(query, candidate_indices, encoded_mentions, graph, target)
            loss /= kwargs['batch_size']
            loss.backward()

            train_loss += loss.item()

            if (batch_idx + 1) % kwargs['batch_size'] == 0:
                # Gradient clipping
                if kwargs['grad_norm']:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                   max_norm=kwargs['grad_norm'])
                optimizer.step()
                optimizer.zero_grad()

                train_steps += 1

                pbar.set_postfix(loss=f'{train_loss / train_steps:.4f}')

            batch_idx += 1

        optimizer.zero_grad()
        pbar.close()

        # Eval loop
        if (epoch + 1) % kwargs['evaluation_step'] == 0:

            model.eval()

            for partition in dataloaders:

                num_correct = 0
                num_steps = 0

                with torch.no_grad():
                    for index, encoded_mentions, graph, target in dataloaders[partition]:

                        num_steps += 1

                        index = index.item()
                        encoded_mentions = encoded_mentions.to(device)
                        graph = graph.to(device)
                        target = target.to(device)

                        # For empty graphs or graphs too large to fit into memory
                        # make a random guess
                        if graph.shape[-1] == 0 or graph.shape[-1] >= TRAIN_SIZE_THRESHOLD:
                            pred = torch.randint(0, target.shape[-1], (1,)).item()
                            if target.squeeze(0)[pred]:
                                num_correct += 1
                            continue

                        # TODO: This is ugly, model itself should deal with the batch index?
                        encoded_mentions = encoded_mentions.squeeze(0)
                        graph = graph.squeeze(0)

                        query = processed_dataset[partition][index]['query']
                        candidate_indices = processed_dataset[partition][index]['candidate_indices']

                        logits, _ = model(query, candidate_indices, encoded_mentions, graph, target)

                        pred = torch.argmax(logits).item()
                        if target.squeeze(0)[pred]:
                            num_correct += 1

                accuracy = num_correct / num_steps

                evaluation_scores[partition][epoch] = accuracy

                if partition == 'dev' and accuracy > best_dev_acc:
                    best_dev_acc = accuracy
                    best_dev_epoch = epoch

                print(f'{partition.title()} accuracy: {accuracy:.2%}')

    print(f'Best dev accuracy was {best_dev_acc:.2%} on epoch: {best_dev_epoch}')

    return evaluation_scores
