import random
import typing as t

import numpy as np
import torch


def set_seed(seed: int, device: t.Union[torch.device, str]) -> None:
    """
    Set seeds for reproducibility.

    :param seed: An integer value to define the seed for random generator.
    :param device: The selected device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # if device.type == 'cuda':
    #     # Also set the deterministic flag for reproducibility
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
        