# Placeholder for utils/helpers.py
# utils/helpers.py

import numpy as np
import torch
import random
import time

def set_global_seed(seed=42):
    """
    Set seed for reproducibility across NumPy, PyTorch, and random.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def measure_time(func):
    """
    Decorator to measure execution time of any function.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution Time: {end - start:.2f} seconds")
        return result
    return wrapper
