import torch
import tqdm


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


# TODO: This code only computes the loss, and does not currently return a prediction
def train(model, processed_dataset, dataloaders, epochs=20):
    """
    """
    device, n_gpus = get_device(model)

    # Cast to list so that we can index in
    processed_dataset = list(processed_dataset.values())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        nb_train_steps = 0

        pbar_descr = 'Epoch: {}/{}'.format(epoch, epochs)
        pbar_train = tqdm(dataloaders['dev'], unit='batch', desc=pbar_descr)

        for _, batch in enumerate(pbar_train):

            model.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            index, encoded_mentions, graph, target = batch

            query = processed_dataset[index]['query']
            candidate_indices = processed_dataset[index]['candidate_indices']

            loss = model(query, candidate_indices, encoded_mentions, graph, target)
            loss.backwards()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=grad_norm)

            # TODO: Need to define the optimizer
            optimizer.step()

            # Loss object is a vector of size n_gpus, need to average if more than 1
            if n_gpus > 1:
                loss = loss.mean()

            # Track train loss
            train_loss += loss.item()
            nb_train_steps += 1

            pbar_train.set_postfix(eval_loss=train_loss / nb_train_steps)

        # Run a validation step at the end of each epoch IF user provided dev partition
        if 'dev' in dataloaders:

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0

            pbar_descr = 'Epoch: {}/{}'.format(epoch, epochs)
            pbar_eval = tqdm(dataloaders['dev'], unit='batch', desc=pbar_descr)

            with torch.no_grad():
                for _, batch in enumerate(pbar_eval):

                    batch = tuple(t.to(device) for t in batch)
                    index, encoded_mentions, graph, target = batch

                    query = processed_dataset[index]['query']
                    candidate_indices = processed_dataset[index]['candidate_indices']

                    loss = model(query, candidate_indices, encoded_mentions, graph, target)

                    if n_gpus > 1:
                        loss = loss.mean()

                    eval_loss += loss.item()
                    nb_eval_steps += 1

                pbar_eval.set_postfix(eval_loss=eval_loss / nb_eval_steps)

        pbar_train.close()
        pbar_eval.close()
