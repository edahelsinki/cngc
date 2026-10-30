import random
import numpy as np
import torch


def get_seeds(n_seeds):
    n_seeds = min(n_seeds, 80)
    seeds = [42, 69, 71, 12, 7, 123, 456, 789, 812, 947,
             1045, 27, 547, 672, 255, 444, 89, 317, 188, 743,
             10, 38, 56, 94, 101, 1123, 1275, 1393, 1459, 1586
             ][:n_seeds]
    return seeds


def fix_seed(seed):
    """
    Fixes the random seed for reproducibility.
    :param seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_fmt(input_value):
    fmt = f"{input_value:.4f}" if input_value > 0.0001 else f"{input_value:.4e}"
    return fmt
