import torch


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
