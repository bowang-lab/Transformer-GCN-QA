import torch
from tqdm import tqdm
# from torch import autograd


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
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
        model_names = ', '.join([torch.cuda.get_device_name(i) for i in range(n_gpu)])
        print('Using CUDA device(s) with name(s): {}.'.format(model_names))
    else:
        device = torch.device("cpu")
        print('No GPU available. Using CPU.')

    return device, n_gpu


# TODO: There is a lot of repeated code between train / dev loops. Encapsulate
def train(model, optimizer, processed_dataset, dataloaders, **kwargs):
    """Trains an instance of `model`.

    Args:
        model (): TODO.
        optimzer (): TODO.
        processed_dataset (): TODO.
        dataloaders (): TODO.
    """
    def train():
        model.train()

        train_loss = 0
        train_acc = 0
        train_steps = 0

        pbar_descr = 'Epoch: {}/{}'.format(epoch, kwargs['epochs'] + 1)
        pbar_train = tqdm(dataloaders['train'], unit='batch', desc=pbar_descr)

        for _, batch in enumerate(pbar_train):

            optimizer.zero_grad()

            index, encoded_mentions, graph, target = batch

            index = index.item()
            encoded_mentions = encoded_mentions.to(device)
            graph = graph.to(device)
            target = target.to(device)

            # TODO: This is kind of ugly, maybe the model itself should deal with the batch index?
            encoded_mentions = encoded_mentions.squeeze(0)
            graph = graph.squeeze(0)

            query = processed_dataset['train'][index]['query']
            candidate_indices = processed_dataset['train'][index]['candidate_indices']

            logits, loss = model(query, candidate_indices, encoded_mentions, graph, target)
            loss.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=grad_norm)

            optimizer.step()

            # Loss object is a vector of size n_gpus, need to average if more than 1
            if n_gpus > 1:
                loss = loss.mean()

            # Track train loss
            train_loss += loss.item()
            train_steps += 1

            # Track train acc
            pred = torch.argmax(logits).item()
            if target.squeeze(0)[pred]:
                train_acc += 1

            pbar_train.set_postfix(loss=train_loss / train_steps)

        pbar_train.close()

        train_loss /= train_steps
        train_acc /= train_steps

        return train_loss, train_acc

    def evaluate():
        model.eval()

        dev_loss = 0
        dev_acc = 0
        dev_steps = 0

        with torch.no_grad():
            for batch in dataloaders['dev']:

                index, encoded_mentions, graph, target = batch

                index = index.item()
                encoded_mentions = encoded_mentions.to(device)
                graph = graph.to(device)
                target = target.to(device)

                # TODO: This is kind of ugly, maybe the model itself should deal with the batch
                # index?
                encoded_mentions = encoded_mentions.squeeze(0)
                graph = graph.squeeze(0)

                query = processed_dataset['dev'][index]['query']
                candidate_indices = processed_dataset['dev'][index]['candidate_indices']

                logits, loss = model(query, candidate_indices, encoded_mentions, graph, target)

                if n_gpus > 1:
                    loss = loss.mean()

                dev_loss += loss.item()
                dev_steps += 1

                pred = torch.argmax(logits).item()
                if target.squeeze(0)[pred]:
                    dev_acc += 1

        dev_loss /= dev_steps
        dev_acc /= dev_steps

        return dev_loss, dev_acc

    device, n_gpus = get_device(model)

    # Cast to list so that we can index in
    processed_dataset = {partition: list(processed_dataset[partition].values())
                         for partition in processed_dataset}

    for epoch in range(kwargs['epochs']):
        # Uncomment to exit and print traceback when NaN appears in backward pass
        # with autograd.detect_anomaly():
        train_loss, train_acc = train()
        print('Train loss:', train_loss)
        print('Train acc:', train_acc)
        # Run a validation step at the end of each epoch IF user provided dev partition
        if 'dev' in processed_dataset:
            dev_loss, dev_acc = evaluate()
            print('Dev loss:', dev_loss)
            print('Dev acc:', dev_acc)
