"""Utility functions for running custom functions in execution."""
# Native Python packages
import random
import string

# Import packages that need to have seeds set
import torch
import numpy as np


def get_random_string(length):
    """Function to generate a random string to ensure we don't run into issues
     running simultaneous ray sessions.

     Parameters:
         length (int): The length of the string we wish to create.

     Returns:
         result_str (str): The randomly-generated string of length given
            by the length parameter.
     """
    letters = string.ascii_lowercase  # String consideration space
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def set_seeds(seed):
    """Function to set seeds for packages to ensure results are reproducible."""
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random library


def get_cuda_objects():
    """Utility function to get cuda objects in memory."""
    import gc
    i = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                    hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
        i += 1
    print("TOTAL CUDA OBJECTS: {}".format(i))
