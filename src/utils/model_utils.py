import gc

import torch
from tqdm import tqdm

from ..constants import TRAIN_SIZE_THRESHOLD


def get_device(model=None):
    """Returns two-tuple containing a PyTorch device (CPU or GPU(s)), and number of available GPUs.

    Returns a two-tuple containing a  PyTorch device (CPU or GPU(s)) and number of available CUDA
    devices. If `model` is not None, and a CUDA device is available, the model is placed on the
    CUDA device with `model.to(device)`. If multiple GPUs are available, the model is parallized
    with `torch.nn.DataParallel(model)`.

    Args:
        (Torch.nn.Module) PyTorch model, if CUDA device is available this function will place the
        model on the CUDA device with `model.to(device)`. If multiple CUDA devices are available,
        the model is parallized with `torch.nn.DataParallel(model)`.

    Returns:
        A two-tuple containing a PyTorch device (CPU or GPU(s)), and number of available GPUs.
    """
    n_gpu = 0

    # use a GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        # if model is provided, we place it on the GPU and parallize it (if possible)
        if model:
            model = model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
        model_names = ', '.join([torch.cuda.get_device_name(i) for i in range(n_gpu)])
        print('Using CUDA device(s) with name(s): {}.'.format(model_names))
    else:
        device = torch.device("cpu")
        print('No GPU available. Using CPU.')

    if model:
        return device, n_gpu, model

    return device, n_gpu


# TODO: Check that this actually helps, otherwise chuck it
def free_memory(*args):
    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()


# TODO: There is a lot of repeated code between train / dev loops. Encapsulate
def train(model, optimizer, processed_dataset, dataloaders, **kwargs):
    """Trains an instance of `model`.

    Args:
        model (): TODO.
        optimizer (torch.optim): TODO.
        processed_dataset (dict): TODO.
        dataloaders (dict): TODO.
    """
    device, n_gpus, model = get_device(model)

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

            # TODO: This is kind of ugly, maybe the model itself should deal with the batch index?
            encoded_mentions = encoded_mentions.squeeze(0)
            graph = graph.squeeze(0)

            # Don't train on empty graphs or those which exceed some size threshold
            if graph.shape[-1] == 0 or graph.shape[-1] >= TRAIN_SIZE_THRESHOLD:
                # print(f'Found a large graph of size {graph.shape[-1]}. Skipping.')
                continue

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

        free_memory(encoded_mentions, graph, target)

        optimizer.zero_grad()
        pbar.close()

        # Eval loop
        if (epoch + 1) % 1 == 0:

            model.eval()

            for partition in dataloaders:

                num_correct = 0
                num_steps = 0

                with torch.no_grad():
                    for index, encoded_mentions, graph, target in dataloaders[partition]:

                        index = index.item()
                        encoded_mentions = encoded_mentions.to(device)
                        graph = graph.to(device)
                        target = target.to(device)

                        # TODO: This is kind of ugly, maybe the model itself should deal with the
                        # batch index?
                        encoded_mentions = encoded_mentions.squeeze(0)
                        graph = graph.squeeze(0)

                        if graph.shape[-1] == 0 or graph.shape[-1] >= TRAIN_SIZE_THRESHOLD:
                            # print(f'Found a large graph of size {graph.shape[-1]}. Skipping.')
                            continue

                        # TODO: Don't do a forward pass on empty graphs, just skip them.
                        query = processed_dataset[partition][index]['query']
                        candidate_indices = processed_dataset[partition][index]['candidate_indices']

                        logits, _ = model(query, candidate_indices, encoded_mentions, graph, target)

                        num_steps += 1

                        pred = torch.argmax(logits).item()
                        if target.squeeze(0)[pred]:
                            num_correct += 1

                accuracy = num_correct / num_steps

                evaluation_scores[partition][epoch] = accuracy

                if partition == 'dev' and accuracy > best_dev_acc:
                    best_dev_acc = accuracy
                    best_dev_epoch = epoch

                print(f'{partition.title()} accuracy: {accuracy:.2%}')

        free_memory(encoded_mentions, graph, target)

    print(f'Best dev accuracy was {best_dev_acc:.2%} on epoch: {best_dev_epoch}')

    return evaluation_scores
