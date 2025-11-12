import logging
import random
import typing as t

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def get_torch_device_name(device: str) -> torch.device:
    """Device name of torch estimation from the device name selected."""
    if device == 'gpu' or 'cuda' in device:
        return torch.device('cuda:0')
    elif device == 'cpu' or device.startswith('cuda:'):
        return torch.device(device)
    else:
        LOGGER.warning(
            ("The device named " + str(device) + " "
             + ("is not available in PyTorch. "
                "So 'cpu' count is selected by default."))
            )
        return torch.device('cpu')


def set_seed(seed: int, device: torch.device) -> None:
    """
    Set seeds for reproducibility.

    :param seed: An integer value to define the seed for random generator.
    :param device: The selected device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if device.type == 'cuda':
        # Also set the deterministic flag for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        