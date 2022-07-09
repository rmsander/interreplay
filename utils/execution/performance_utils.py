"""Utils for improving performance speed."""
# External Python packages
import torch


def determine_device(gpu_devices, framework, cpu_only=False):
    """Function for setting the context manager for running with devices.

    Parameters:
        gpu_devices (list): A list of strings corresponding to the GPU
            devices available for use.
        framework (str): A string denoting the deep learning framework (either
            'tf' for TensorFlow or 'torch' for PyTorch.
        cpu_only (bool): Whether to explicitly use only the CPU.

    Returns:
        device (str): A string corresponding to the device to be used.
    """
    # Check if using tensorflow
    if framework == "tf" or framework == "tf2":
        # Check if we have gpu(s) available
        if len(gpu_devices) > 0:  # Use GPU
            device = gpu_devices[0]
        else:  # Use CPU
            device = 'cpu'

    # Check if using torch
    if framework == "torch":
        if len(gpu_devices) > 0:  # Use GPU
            device = torch.device('cuda:0')
        else:  # Use CPU
            device = 'cpu'

    # Override if using cpu only
    if cpu_only:
        device = 'cpu'

    # Use device for interpolating batches
    return device


def determine_device_torch(cpu_only=False):
    """Utility function to return the device for torch.

    Parameters:
        cpu_only (bool): Whether to exclusively use the CPU.  Defaults to False.

    Returns:
        device (torch.device or string): A device or string object corresponding
            to the device to be used for torch.
    """
    if torch.cuda.is_available() and not cpu_only:  # Use GPU
        device = torch.device('cuda:0')  # TODO(rms): If multi-GPU, generalizse
    else:  # Use CPU
        device = 'cpu'
    return device
